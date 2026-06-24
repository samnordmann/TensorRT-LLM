# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental CuTe bf16 allreduce + residual + RMSNorm kernel.

This module is intentionally narrow and supports eager experimentation only.
The prototype uses PyTorch symmetric-memory barriers around the custom kernel.
CUDA graph capture is outside this path's supported envelope until allocation
and compilation are moved out of the callable path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith, llvm, vector
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

_DEFAULT_TPC_VALUES = (64, 128, 256, 512, 1024)
_SYNC_MODE_INTERNAL = "internal"
_SYNC_MODE_CALLER = "caller"
_SYNC_MODES = (_SYNC_MODE_INTERNAL, _SYNC_MODE_CALLER)


@dataclass
class _SymmTensor:
    tensor: torch.Tensor
    handle: object

    @property
    def multicast_ptr(self) -> int | None:
        try:
            ptr = self.handle.multicast_ptr
        except RuntimeError:
            return None
        return int(ptr) if ptr else None


@dataclass
class _Runtime:
    symm_input: _SymmTensor
    launch: Callable
    threads_per_cta: int


@dataclass
class _TwoShotRuntime:
    symm_input: _SymmTensor
    symm_reduced: _SymmTensor
    reduce_broadcast_launch: Callable
    rmsnorm_launch: Callable
    reduce_threads_per_cta: int
    rmsnorm_threads_per_cta: int


@dataclass
class _FusedTwoShotRuntime:
    symm_input: _SymmTensor
    symm_output: _SymmTensor
    symm_sync: _SymmTensor | None
    launch: Callable
    threads_per_cta: int


@dataclass
class _TileFusedTwoShotRuntime:
    symm_input: _SymmTensor
    symm_output: _SymmTensor
    symm_partials: _SymmTensor
    symm_sync: _SymmTensor
    launch: Callable
    threads_per_cta: int
    worker_ctas: int


_RUNTIME_CACHE: dict[tuple[int, int, int, torch.dtype], _Runtime] = {}
_TWOSHOT_RUNTIME_CACHE: dict[
    tuple[int, int, int, torch.dtype, int, int, int, int, bool],
    _TwoShotRuntime,
] = {}
_FUSED_TWOSHOT_RUNTIME_CACHE: dict[
    tuple[int, int, int, torch.dtype, int, int, int, bool],
    _FusedTwoShotRuntime,
] = {}
_TILE_FUSED_TWOSHOT_RUNTIME_CACHE: dict[
    tuple[int, int, int, torch.dtype, int, int, int, int],
    _TileFusedTwoShotRuntime,
] = {}
_REGISTERED_SYMM_INPUTS: dict[tuple[int, int, int, torch.dtype], _SymmTensor] = {}


def _require_cute() -> None:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise NotImplementedError("CuTe DSL is not available")


def _current_cu_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _valid_threads_per_cta(hidden_size: int) -> int:
    valid = _valid_threads_per_cta_values(hidden_size)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    override = os.environ.get("TRTLLM_CUTE_AR_RMSNORM_THREADS_PER_CTA")
    if override is not None:
        return _validate_threads_per_cta_override(hidden_size, override, valid)
    return valid[-1]


def _valid_threads_per_cta_values(hidden_size: int) -> tuple[int, ...]:
    return tuple(tpc for tpc in _DEFAULT_TPC_VALUES if hidden_size % (tpc * 8) == 0)


def _validate_threads_per_cta_override(
    hidden_size: int,
    override: str,
    valid: tuple[int, ...],
) -> int:
    threads_per_cta = int(override)
    if threads_per_cta not in valid:
        raise NotImplementedError(
            f"threads_per_cta={threads_per_cta} is not supported for hidden_size={hidden_size}"
        )
    return threads_per_cta


def _twoshot_threads_per_cta_override(name: str) -> str | None:
    override = os.environ.get(name)
    if override is not None:
        return override
    return os.environ.get("TRTLLM_CUTE_AR_RMSNORM_THREADS_PER_CTA")


def _valid_twoshot_reduce_threads_per_cta(
    rows: int,
    hidden_size: int,
    world_size: int,
) -> int:
    override = _twoshot_threads_per_cta_override(
        "TRTLLM_CUTE_AR_RMSNORM_TWOSHOT_REDUCE_THREADS_PER_CTA"
    )
    valid = _valid_threads_per_cta_values(hidden_size)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    if override is not None:
        return _validate_threads_per_cta_override(hidden_size, override, valid)

    if rows >= 1024 and 128 in valid:
        return 128
    if rows >= 512 and 256 in valid:
        return 256
    if rows >= 256 and 512 in valid:
        return 512

    min_chunks = max(1, (world_size + max(rows, 1) - 1) // max(rows, 1))
    candidates = tuple(tpc for tpc in valid if hidden_size // (tpc * 8) >= min_chunks)
    if candidates:
        return candidates[-1]
    return valid[0]


def _valid_twoshot_rmsnorm_threads_per_cta(rows: int, hidden_size: int) -> int:
    override = _twoshot_threads_per_cta_override(
        "TRTLLM_CUTE_AR_RMSNORM_TWOSHOT_RMSNORM_THREADS_PER_CTA"
    )
    valid = _valid_threads_per_cta_values(hidden_size)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    if override is not None:
        return _validate_threads_per_cta_override(hidden_size, override, valid)

    if rows >= 1024 and hidden_size >= 16384 and 1024 in valid:
        return 1024
    if rows >= 512 and 512 in valid:
        return 512
    if rows >= 256 and 512 in valid:
        return 512
    return valid[-1]


def _valid_fused_twoshot_threads_per_cta(rows: int, hidden_size: int) -> int:
    override = _twoshot_threads_per_cta_override(
        "TRTLLM_CUTE_AR_RMSNORM_FUSED_TWOSHOT_THREADS_PER_CTA"
    )
    valid = _valid_threads_per_cta_values(hidden_size)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    if override is not None:
        return _validate_threads_per_cta_override(hidden_size, override, valid)

    if rows >= 512 and hidden_size >= 8192 and 256 in valid:
        return 256
    return _valid_twoshot_rmsnorm_threads_per_cta(rows, hidden_size)


def _valid_tile_fused_twoshot_threads_per_cta(hidden_size: int) -> int:
    override = _twoshot_threads_per_cta_override(
        "TRTLLM_CUTE_AR_RMSNORM_TILE_FUSED_TWOSHOT_THREADS_PER_CTA"
    )
    valid = _valid_threads_per_cta_values(hidden_size)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    if override is not None:
        return _validate_threads_per_cta_override(hidden_size, override, valid)
    return 128 if 128 in valid else valid[0]


def _tile_fused_twoshot_worker_ctas(owned_tiles: int) -> int:
    override = os.environ.get("TRTLLM_CUTE_AR_RMSNORM_TILE_FUSED_TWOSHOT_WORKERS")
    limit = int(override) if override is not None else 64
    return min(max(1, owned_tiles), max(1, limit))


def _use_twoshot_tile_ownership(rows: int, world_size: int) -> bool:
    override = os.environ.get("TRTLLM_CUTE_AR_RMSNORM_TWOSHOT_TILE_OWNERSHIP")
    if override is not None:
        return override == "1"
    return rows < world_size


def _enable_symmetric_memory() -> None:
    enable_fn = getattr(symm_mem, "enable_symm_mem_for_group", None)
    if enable_fn is not None:
        try:
            enable_fn(dist.group.WORLD.group_name)
        except RuntimeError:
            pass


def _alloc_symm_tensor(numel: int, dtype: torch.dtype) -> _SymmTensor:
    tensor = symm_mem.empty(numel, dtype=dtype, device="cuda")
    handle = symm_mem.rendezvous(tensor, dist.group.WORLD)
    return _SymmTensor(tensor=tensor, handle=handle)


def _symm_input_key(tensor: torch.Tensor) -> tuple[int, int, int, torch.dtype]:
    return (
        torch.cuda.current_device(),
        int(tensor.data_ptr()),
        int(tensor.numel()),
        tensor.dtype,
    )


def allocate_symmetric_tensor_like(reference: torch.Tensor) -> torch.Tensor:
    """Allocate and register a symmetric tensor usable as no-copy CuTe input.

    The returned tensor is shaped like ``reference``. A producer can write
    directly into it, then pass it to ``fused_allreduce_residual_rmsnorm_bf16``
    without the POC staging copy.
    """
    _require_cute()
    if not dist.is_initialized():
        raise NotImplementedError("torch.distributed must be initialized")
    if not reference.is_cuda:
        raise NotImplementedError("symmetric tensor allocation requires a CUDA reference")

    _enable_symmetric_memory()
    symm_input = _alloc_symm_tensor(reference.numel(), reference.dtype)
    _REGISTERED_SYMM_INPUTS[_symm_input_key(symm_input.tensor)] = symm_input
    return symm_input.tensor.view_as(reference)


def _lookup_symmetric_input(tensor: torch.Tensor) -> _SymmTensor | None:
    return _REGISTERED_SYMM_INPUTS.get(_symm_input_key(tensor.reshape(-1)))


def _validate_sync_mode(sync_mode: str) -> str:
    sync_mode = sync_mode.lower()
    if sync_mode not in _SYNC_MODES:
        raise NotImplementedError(
            f"unknown synchronization mode {sync_mode!r}; expected one of {_SYNC_MODES}"
        )
    return sync_mode


def _check_common_inputs(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> int:
    if not dist.is_initialized():
        raise NotImplementedError("torch.distributed must be initialized")
    if input.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise NotImplementedError("only bf16 activations are supported")
    if weight.dtype != torch.bfloat16:
        raise NotImplementedError("only bf16 RMSNorm weight is supported")
    if input.shape != residual.shape:
        raise NotImplementedError("input and residual shapes must match")
    if weight.ndim != 1:
        raise NotImplementedError("RMSNorm weight must be 1D")

    hidden_size = weight.numel()
    if input.numel() % hidden_size != 0:
        raise NotImplementedError("activation size is not divisible by hidden size")
    return hidden_size


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _raw_tensor_1d(addr: Int64, dtype, size):
        ptr = cute.make_ptr(dtype, addr, cute.AddressSpace.gmem)
        return cute.make_tensor(ptr, cute.make_layout((size,), stride=(1,)))

    @cute.jit
    def _raw_tensor_2d(addr: Int64, dtype, rows, cols):
        ptr = cute.make_ptr(dtype, addr, cute.AddressSpace.gmem)
        return cute.make_tensor(ptr, cute.make_layout((rows, cols), stride=(cols, 1)))

    @dsl_user_op
    def _i32_to_bf16_pair(x, *, loc=None, ip=None):
        x_ir = x.ir_value(loc=loc, ip=ip) if hasattr(x, "ir_value") else x
        vec_ty = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
        vec = llvm.bitcast(vec_ty, x_ir, loc=loc, ip=ip)
        lo = vector.extractelement(
            vec,
            position=arith.constant(Int32.mlir_type, 0, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        hi = vector.extractelement(
            vec,
            position=arith.constant(Int32.mlir_type, 1, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        return BFloat16(lo), BFloat16(hi)

    @dsl_user_op
    def _ld_global_8xbf16(uc_ptr: cute.Pointer, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), uc_ptr.llvm_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
            [ptr_i64],
            "ld.global.v4.u32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
        r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
        r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
        r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
        return Int32(r0), Int32(r1), Int32(r2), Int32(r3)

    @dsl_user_op
    def _f32x2_to_bf16x2(a, b, *, loc=None, ip=None):
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
                "cvt.rn.bf16x2.f32 $0, $2, $1;",
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _st_global_8xbf16(uc_ptr: cute.Pointer, r0, r1, r2, r3, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), uc_ptr.llvm_ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                ptr_i64,
                Int32(r0).ir_value(loc=loc, ip=ip),
                Int32(r1).ir_value(loc=loc, ip=ip),
                Int32(r2).ir_value(loc=loc, ip=ip),
                Int32(r3).ir_value(loc=loc, ip=ip),
            ],
            "st.global.v4.u32 [$0], {$1,$2,$3,$4};",
            "l,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _multimem_st_8xbf16(mc_ptr: cute.Pointer, r0, r1, r2, r3, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), mc_ptr.llvm_ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                ptr_i64,
                Int32(r0).ir_value(loc=loc, ip=ip),
                Int32(r1).ir_value(loc=loc, ip=ip),
                Int32(r2).ir_value(loc=loc, ip=ip),
                Int32(r3).ir_value(loc=loc, ip=ip),
            ],
            "multimem.st.global.v4.f32 [$0], {$1,$2,$3,$4};",
            "l,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_global_f32(uc_ptr: cute.Pointer, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), uc_ptr.llvm_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.f32(),
            [ptr_i64],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return Float32(result)

    @dsl_user_op
    def _multimem_st_f32(mc_ptr: cute.Pointer, value, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), mc_ptr.llvm_ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                ptr_i64,
                Float32(value).ir_value(loc=loc, ip=ip),
            ],
            "multimem.st.global.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_global_i32_volatile(uc_ptr: cute.Pointer, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), uc_ptr.llvm_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.i32(),
            [ptr_i64],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return Int32(result)

    @dsl_user_op
    def _atom_global_add_i32(uc_ptr: cute.Pointer, value, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), uc_ptr.llvm_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.i32(),
            [
                ptr_i64,
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return Int32(result)

    @dsl_user_op
    def _multimem_st_i32(mc_ptr: cute.Pointer, value, *, loc=None, ip=None):
        ptr_i64 = llvm.ptrtoint(T.i64(), mc_ptr.llvm_ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                ptr_i64,
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "multimem.st.global.u32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _fence_acq_rel_sys(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "fence.acq_rel.sys;",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _spin_pause(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    class _FusedArRmsnormBf16Launcher:
        def __init__(
            self,
            rows: int,
            hidden_size: int,
            threads_per_cta: int,
        ):
            self.rows = rows
            self.hidden_size = hidden_size
            self.threads_per_cta = threads_per_cta
            self.vec_chunks = hidden_size // (threads_per_cta * 8)

        @cute.kernel
        def kernel(
            self,
            mc_in: cute.Tensor,
            uc_residual: cute.Tensor,
            uc_weight: cute.Tensor,
            uc_h_out: cute.Tensor,
            uc_y_out: cute.Tensor,
            eps: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()

            frag_h = cute.make_rmem_tensor((self.vec_chunks, 8), Float32)
            sum_sq = Float32(0.0)

            for c in cutlass.range_constexpr(self.vec_chunks):
                col = c * self.threads_per_cta * 8 + tidx * 8
                offset = row * self.hidden_size + col

                r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(
                    mc_in.iterator + offset
                )
                b0, b1 = _i32_to_bf16_pair(r0)
                b2, b3 = _i32_to_bf16_pair(r1)
                b4, b5 = _i32_to_bf16_pair(r2)
                b6, b7 = _i32_to_bf16_pair(r3)

                res_r0, res_r1, res_r2, res_r3 = _ld_global_8xbf16(
                    uc_residual.iterator + offset
                )
                res0, res1 = _i32_to_bf16_pair(res_r0)
                res2, res3 = _i32_to_bf16_pair(res_r1)
                res4, res5 = _i32_to_bf16_pair(res_r2)
                res6, res7 = _i32_to_bf16_pair(res_r3)

                h0 = Float32(b0) + Float32(res0)
                h1 = Float32(b1) + Float32(res1)
                h2 = Float32(b2) + Float32(res2)
                h3 = Float32(b3) + Float32(res3)
                h4 = Float32(b4) + Float32(res4)
                h5 = Float32(b5) + Float32(res5)
                h6 = Float32(b6) + Float32(res6)
                h7 = Float32(b7) + Float32(res7)

                _st_global_8xbf16(
                    uc_h_out.iterator + offset,
                    _f32x2_to_bf16x2(h0, h1),
                    _f32x2_to_bf16x2(h2, h3),
                    _f32x2_to_bf16x2(h4, h5),
                    _f32x2_to_bf16x2(h6, h7),
                )

                sum_sq = (
                    sum_sq
                    + h0 * h0
                    + h1 * h1
                    + h2 * h2
                    + h3 * h3
                    + h4 * h4
                    + h5 * h5
                    + h6 * h6
                    + h7 * h7
                )

                frag_h[c, 0] = h0
                frag_h[c, 1] = h1
                frag_h[c, 2] = h2
                frag_h[c, 3] = h3
                frag_h[c, 4] = h4
                frag_h[c, 5] = h5
                frag_h[c, 6] = h6
                frag_h[c, 7] = h7

            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            num_warps = self.threads_per_cta // 32

            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            smem_ptr = cute.arch.alloc_smem(Float32, num_warps + 1)
            smem = cute.make_tensor(smem_ptr, cute.make_layout((num_warps + 1,)))

            if lane_idx == 0:
                smem[warp_idx] = sum_sq
            cute.arch.barrier()

            if warp_idx == 0:
                part = Float32(0.0)
                if lane_idx < num_warps:
                    part = smem[lane_idx]
                part = cute.arch.warp_reduction_sum(part)
                if lane_idx == 0:
                    smem[num_warps] = cute.math.rsqrt(
                        part / Float32(self.hidden_size) + eps,
                        fastmath=True,
                    )
            cute.arch.barrier()

            inv_rms = smem[num_warps]

            for c in cutlass.range_constexpr(self.vec_chunks):
                col = c * self.threads_per_cta * 8 + tidx * 8
                offset = row * self.hidden_size + col

                w_r0, w_r1, w_r2, w_r3 = _ld_global_8xbf16(uc_weight.iterator + col)
                w0, w1 = _i32_to_bf16_pair(w_r0)
                w2, w3 = _i32_to_bf16_pair(w_r1)
                w4, w5 = _i32_to_bf16_pair(w_r2)
                w6, w7 = _i32_to_bf16_pair(w_r3)

                y0 = frag_h[c, 0] * inv_rms * Float32(w0)
                y1 = frag_h[c, 1] * inv_rms * Float32(w1)
                y2 = frag_h[c, 2] * inv_rms * Float32(w2)
                y3 = frag_h[c, 3] * inv_rms * Float32(w3)
                y4 = frag_h[c, 4] * inv_rms * Float32(w4)
                y5 = frag_h[c, 5] * inv_rms * Float32(w5)
                y6 = frag_h[c, 6] * inv_rms * Float32(w6)
                y7 = frag_h[c, 7] * inv_rms * Float32(w7)

                _st_global_8xbf16(
                    uc_y_out.iterator + offset,
                    _f32x2_to_bf16x2(y0, y1),
                    _f32x2_to_bf16x2(y2, y3),
                    _f32x2_to_bf16x2(y4, y5),
                    _f32x2_to_bf16x2(y6, y7),
                )

        @cute.jit
        def __call__(
            self,
            mc_addr_in: Int64,
            uc_addr_residual: Int64,
            uc_addr_weight: Int64,
            uc_addr_h_out: Int64,
            uc_addr_y_out: Int64,
            eps: Float32,
            stream: cuda.CUstream,
        ):
            mc_in = _raw_tensor_2d(mc_addr_in, BFloat16, self.rows, self.hidden_size)
            uc_res = _raw_tensor_2d(uc_addr_residual, BFloat16, self.rows, self.hidden_size)
            uc_w = _raw_tensor_1d(uc_addr_weight, BFloat16, self.hidden_size)
            uc_h = _raw_tensor_2d(uc_addr_h_out, BFloat16, self.rows, self.hidden_size)
            uc_y = _raw_tensor_2d(uc_addr_y_out, BFloat16, self.rows, self.hidden_size)

            self.kernel(
                mc_in,
                uc_res,
                uc_w,
                uc_h,
                uc_y,
                eps,
            ).launch(
                grid=[self.rows, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )

    class _TwoShotReduceBroadcastBf16Launcher:
        def __init__(
            self,
            rows: int,
            hidden_size: int,
            threads_per_cta: int,
            world_size: int,
            rank: int,
            tile_ownership: bool,
        ):
            self.rows = rows
            self.hidden_size = hidden_size
            self.threads_per_cta = threads_per_cta
            self.world_size = world_size
            self.rank = rank
            self.tile_ownership = tile_ownership
            self.vec_chunks = hidden_size // (threads_per_cta * 8)
            self.total_tiles = rows * self.vec_chunks
            self.owned_rows = max(
                1,
                (rows + world_size - 1 - rank) // world_size if rank < rows else 0,
            )
            self.owned_tiles = max(
                1,
                (self.total_tiles + world_size - 1 - rank) // world_size
                if rank < self.total_tiles
                else 0,
            )

        @cute.kernel
        def kernel(
            self,
            mc_in: cute.Tensor,
            mc_reduced: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            owner_idx, _, _ = cute.arch.block_idx()

            if self.tile_ownership:
                tile_id = owner_idx * self.world_size + self.rank
                if tile_id < self.total_tiles:
                    row = tile_id // self.vec_chunks
                    chunk = tile_id % self.vec_chunks
                    col = chunk * self.threads_per_cta * 8 + tidx * 8
                    offset = row * self.hidden_size + col

                    r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(
                        mc_in.iterator + offset
                    )
                    _multimem_st_8xbf16(mc_reduced.iterator + offset, r0, r1, r2, r3)
            else:
                row = owner_idx * self.world_size + self.rank
                if row < self.rows:
                    for c in cutlass.range_constexpr(self.vec_chunks):
                        col = c * self.threads_per_cta * 8 + tidx * 8
                        offset = row * self.hidden_size + col

                        r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(
                            mc_in.iterator + offset
                        )
                        _multimem_st_8xbf16(mc_reduced.iterator + offset, r0, r1, r2, r3)

        @cute.jit
        def __call__(
            self,
            mc_addr_in: Int64,
            mc_addr_reduced: Int64,
            stream: cuda.CUstream,
        ):
            mc_in = _raw_tensor_2d(mc_addr_in, BFloat16, self.rows, self.hidden_size)
            mc_reduced = _raw_tensor_2d(
                mc_addr_reduced, BFloat16, self.rows, self.hidden_size
            )

            self.kernel(mc_in, mc_reduced).launch(
                grid=[self.owned_tiles if self.tile_ownership else self.owned_rows, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )

    class _TwoShotRmsnormBf16Launcher:
        def __init__(
            self,
            rows: int,
            hidden_size: int,
            threads_per_cta: int,
        ):
            self.rows = rows
            self.hidden_size = hidden_size
            self.threads_per_cta = threads_per_cta
            self.vec_chunks = hidden_size // (threads_per_cta * 8)

        @cute.kernel
        def kernel(
            self,
            uc_reduced: cute.Tensor,
            uc_residual: cute.Tensor,
            uc_weight: cute.Tensor,
            uc_h_out: cute.Tensor,
            uc_y_out: cute.Tensor,
            eps: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()

            frag_h = cute.make_rmem_tensor((self.vec_chunks, 8), Float32)
            sum_sq = Float32(0.0)

            for c in cutlass.range_constexpr(self.vec_chunks):
                col = c * self.threads_per_cta * 8 + tidx * 8
                offset = row * self.hidden_size + col

                red_r0, red_r1, red_r2, red_r3 = _ld_global_8xbf16(
                    uc_reduced.iterator + offset
                )
                red0, red1 = _i32_to_bf16_pair(red_r0)
                red2, red3 = _i32_to_bf16_pair(red_r1)
                red4, red5 = _i32_to_bf16_pair(red_r2)
                red6, red7 = _i32_to_bf16_pair(red_r3)

                res_r0, res_r1, res_r2, res_r3 = _ld_global_8xbf16(
                    uc_residual.iterator + offset
                )
                res0, res1 = _i32_to_bf16_pair(res_r0)
                res2, res3 = _i32_to_bf16_pair(res_r1)
                res4, res5 = _i32_to_bf16_pair(res_r2)
                res6, res7 = _i32_to_bf16_pair(res_r3)

                h0 = Float32(red0) + Float32(res0)
                h1 = Float32(red1) + Float32(res1)
                h2 = Float32(red2) + Float32(res2)
                h3 = Float32(red3) + Float32(res3)
                h4 = Float32(red4) + Float32(res4)
                h5 = Float32(red5) + Float32(res5)
                h6 = Float32(red6) + Float32(res6)
                h7 = Float32(red7) + Float32(res7)

                _st_global_8xbf16(
                    uc_h_out.iterator + offset,
                    _f32x2_to_bf16x2(h0, h1),
                    _f32x2_to_bf16x2(h2, h3),
                    _f32x2_to_bf16x2(h4, h5),
                    _f32x2_to_bf16x2(h6, h7),
                )

                sum_sq = (
                    sum_sq
                    + h0 * h0
                    + h1 * h1
                    + h2 * h2
                    + h3 * h3
                    + h4 * h4
                    + h5 * h5
                    + h6 * h6
                    + h7 * h7
                )

                frag_h[c, 0] = h0
                frag_h[c, 1] = h1
                frag_h[c, 2] = h2
                frag_h[c, 3] = h3
                frag_h[c, 4] = h4
                frag_h[c, 5] = h5
                frag_h[c, 6] = h6
                frag_h[c, 7] = h7

            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            num_warps = self.threads_per_cta // 32

            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            smem_ptr = cute.arch.alloc_smem(Float32, num_warps + 1)
            smem = cute.make_tensor(smem_ptr, cute.make_layout((num_warps + 1,)))

            if lane_idx == 0:
                smem[warp_idx] = sum_sq
            cute.arch.barrier()

            if warp_idx == 0:
                part = Float32(0.0)
                if lane_idx < num_warps:
                    part = smem[lane_idx]
                part = cute.arch.warp_reduction_sum(part)
                if lane_idx == 0:
                    smem[num_warps] = cute.math.rsqrt(
                        part / Float32(self.hidden_size) + eps,
                        fastmath=True,
                    )
            cute.arch.barrier()

            inv_rms = smem[num_warps]

            for c in cutlass.range_constexpr(self.vec_chunks):
                col = c * self.threads_per_cta * 8 + tidx * 8
                offset = row * self.hidden_size + col

                w_r0, w_r1, w_r2, w_r3 = _ld_global_8xbf16(uc_weight.iterator + col)
                w0, w1 = _i32_to_bf16_pair(w_r0)
                w2, w3 = _i32_to_bf16_pair(w_r1)
                w4, w5 = _i32_to_bf16_pair(w_r2)
                w6, w7 = _i32_to_bf16_pair(w_r3)

                y0 = frag_h[c, 0] * inv_rms * Float32(w0)
                y1 = frag_h[c, 1] * inv_rms * Float32(w1)
                y2 = frag_h[c, 2] * inv_rms * Float32(w2)
                y3 = frag_h[c, 3] * inv_rms * Float32(w3)
                y4 = frag_h[c, 4] * inv_rms * Float32(w4)
                y5 = frag_h[c, 5] * inv_rms * Float32(w5)
                y6 = frag_h[c, 6] * inv_rms * Float32(w6)
                y7 = frag_h[c, 7] * inv_rms * Float32(w7)

                _st_global_8xbf16(
                    uc_y_out.iterator + offset,
                    _f32x2_to_bf16x2(y0, y1),
                    _f32x2_to_bf16x2(y2, y3),
                    _f32x2_to_bf16x2(y4, y5),
                    _f32x2_to_bf16x2(y6, y7),
                )

        @cute.jit
        def __call__(
            self,
            uc_addr_reduced: Int64,
            uc_addr_residual: Int64,
            uc_addr_weight: Int64,
            uc_addr_h_out: Int64,
            uc_addr_y_out: Int64,
            eps: Float32,
            stream: cuda.CUstream,
        ):
            uc_reduced = _raw_tensor_2d(
                uc_addr_reduced, BFloat16, self.rows, self.hidden_size
            )
            uc_res = _raw_tensor_2d(uc_addr_residual, BFloat16, self.rows, self.hidden_size)
            uc_w = _raw_tensor_1d(uc_addr_weight, BFloat16, self.hidden_size)
            uc_h = _raw_tensor_2d(uc_addr_h_out, BFloat16, self.rows, self.hidden_size)
            uc_y = _raw_tensor_2d(uc_addr_y_out, BFloat16, self.rows, self.hidden_size)

            self.kernel(
                uc_reduced,
                uc_res,
                uc_w,
                uc_h,
                uc_y,
                eps,
            ).launch(
                grid=[self.rows, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )


    class _FusedTwoShotBf16Launcher:
        def __init__(
            self,
            rows: int,
            hidden_size: int,
            threads_per_cta: int,
            world_size: int,
            rank: int,
            device_sync: bool,
        ):
            self.rows = rows
            self.hidden_size = hidden_size
            self.threads_per_cta = threads_per_cta
            self.world_size = world_size
            self.rank = rank
            self.device_sync = device_sync
            self.vec_chunks = hidden_size // (threads_per_cta * 8)
            self.numel = rows * hidden_size
            self.real_owned_rows = (
                (rows + world_size - 1 - rank) // world_size if rank < rows else 0
            )
            self.owned_rows = max(1, self.real_owned_rows)
            self.grid_rows = self.owned_rows + 1 if device_sync else self.owned_rows

        @cute.kernel
        def kernel(
            self,
            mc_in: cute.Tensor,
            uc_residual: cute.Tensor,
            uc_weight: cute.Tensor,
            mc_output: cute.Tensor,
            uc_sync: cute.Tensor,
            mc_sync: cute.Tensor,
            eps: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            owner_idx, _, _ = cute.arch.block_idx()

            if self.device_sync and owner_idx == self.owned_rows:
                if tidx == 0:
                    epoch = _atom_global_add_i32(uc_sync.iterator + 1, Int32(1)) + Int32(1)
                    target = epoch * Int32(self.real_owned_rows)
                    while _ld_global_i32_volatile(uc_sync.iterator + 0) < target:
                        _spin_pause()

                    _fence_acq_rel_sys()
                    _multimem_st_i32(mc_sync.iterator + 2 + self.rank, epoch)
                    for peer in cutlass.range_constexpr(self.world_size):
                        while _ld_global_i32_volatile(uc_sync.iterator + 2 + peer) < epoch:
                            _spin_pause()
            else:
                row = owner_idx * self.world_size + self.rank
                if owner_idx < self.real_owned_rows:
                    self._compute_row(
                        tidx,
                        row,
                        mc_in,
                        uc_residual,
                        uc_weight,
                        mc_output,
                        eps,
                    )
                    if self.device_sync and tidx == 0:
                        _fence_acq_rel_sys()
                        _atom_global_add_i32(uc_sync.iterator + 0, Int32(1))

        @cute.jit
        def _compute_row(
            self,
            tidx,
            row,
            mc_in: cute.Tensor,
            uc_residual: cute.Tensor,
            uc_weight: cute.Tensor,
            mc_output: cute.Tensor,
            eps: Float32,
        ):
                frag_h = cute.make_rmem_tensor((self.vec_chunks, 8), Float32)
                sum_sq = Float32(0.0)

                for c in cutlass.range_constexpr(self.vec_chunks):
                    col = c * self.threads_per_cta * 8 + tidx * 8
                    offset = row * self.hidden_size + col

                    r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(
                        mc_in.iterator + offset
                    )
                    b0, b1 = _i32_to_bf16_pair(r0)
                    b2, b3 = _i32_to_bf16_pair(r1)
                    b4, b5 = _i32_to_bf16_pair(r2)
                    b6, b7 = _i32_to_bf16_pair(r3)

                    res_r0, res_r1, res_r2, res_r3 = _ld_global_8xbf16(
                        uc_residual.iterator + offset
                    )
                    res0, res1 = _i32_to_bf16_pair(res_r0)
                    res2, res3 = _i32_to_bf16_pair(res_r1)
                    res4, res5 = _i32_to_bf16_pair(res_r2)
                    res6, res7 = _i32_to_bf16_pair(res_r3)

                    h0 = Float32(b0) + Float32(res0)
                    h1 = Float32(b1) + Float32(res1)
                    h2 = Float32(b2) + Float32(res2)
                    h3 = Float32(b3) + Float32(res3)
                    h4 = Float32(b4) + Float32(res4)
                    h5 = Float32(b5) + Float32(res5)
                    h6 = Float32(b6) + Float32(res6)
                    h7 = Float32(b7) + Float32(res7)

                    _multimem_st_8xbf16(
                        mc_output.iterator + offset,
                        _f32x2_to_bf16x2(h0, h1),
                        _f32x2_to_bf16x2(h2, h3),
                        _f32x2_to_bf16x2(h4, h5),
                        _f32x2_to_bf16x2(h6, h7),
                    )

                    sum_sq = (
                        sum_sq
                        + h0 * h0
                        + h1 * h1
                        + h2 * h2
                        + h3 * h3
                        + h4 * h4
                        + h5 * h5
                        + h6 * h6
                        + h7 * h7
                    )

                    frag_h[c, 0] = h0
                    frag_h[c, 1] = h1
                    frag_h[c, 2] = h2
                    frag_h[c, 3] = h3
                    frag_h[c, 4] = h4
                    frag_h[c, 5] = h5
                    frag_h[c, 6] = h6
                    frag_h[c, 7] = h7

                lane_idx = cute.arch.lane_idx()
                warp_idx = cute.arch.warp_idx()
                num_warps = self.threads_per_cta // 32

                sum_sq = cute.arch.warp_reduction_sum(sum_sq)
                smem_ptr = cute.arch.alloc_smem(Float32, num_warps + 1)
                smem = cute.make_tensor(smem_ptr, cute.make_layout((num_warps + 1,)))

                if lane_idx == 0:
                    smem[warp_idx] = sum_sq
                cute.arch.barrier()

                if warp_idx == 0:
                    part = Float32(0.0)
                    if lane_idx < num_warps:
                        part = smem[lane_idx]
                    part = cute.arch.warp_reduction_sum(part)
                    if lane_idx == 0:
                        smem[num_warps] = cute.math.rsqrt(
                            part / Float32(self.hidden_size) + eps,
                            fastmath=True,
                        )
                cute.arch.barrier()

                inv_rms = smem[num_warps]

                for c in cutlass.range_constexpr(self.vec_chunks):
                    col = c * self.threads_per_cta * 8 + tidx * 8
                    offset = row * self.hidden_size + col

                    w_r0, w_r1, w_r2, w_r3 = _ld_global_8xbf16(uc_weight.iterator + col)
                    w0, w1 = _i32_to_bf16_pair(w_r0)
                    w2, w3 = _i32_to_bf16_pair(w_r1)
                    w4, w5 = _i32_to_bf16_pair(w_r2)
                    w6, w7 = _i32_to_bf16_pair(w_r3)

                    y0 = frag_h[c, 0] * inv_rms * Float32(w0)
                    y1 = frag_h[c, 1] * inv_rms * Float32(w1)
                    y2 = frag_h[c, 2] * inv_rms * Float32(w2)
                    y3 = frag_h[c, 3] * inv_rms * Float32(w3)
                    y4 = frag_h[c, 4] * inv_rms * Float32(w4)
                    y5 = frag_h[c, 5] * inv_rms * Float32(w5)
                    y6 = frag_h[c, 6] * inv_rms * Float32(w6)
                    y7 = frag_h[c, 7] * inv_rms * Float32(w7)

                    _multimem_st_8xbf16(
                        mc_output.iterator + self.numel + offset,
                        _f32x2_to_bf16x2(y0, y1),
                        _f32x2_to_bf16x2(y2, y3),
                        _f32x2_to_bf16x2(y4, y5),
                        _f32x2_to_bf16x2(y6, y7),
                    )

        @cute.jit
        def __call__(
            self,
            mc_addr_in: Int64,
            uc_addr_residual: Int64,
            uc_addr_weight: Int64,
            mc_addr_output: Int64,
            uc_addr_sync: Int64,
            mc_addr_sync: Int64,
            eps: Float32,
            stream: cuda.CUstream,
        ):
            mc_in = _raw_tensor_2d(mc_addr_in, BFloat16, self.rows, self.hidden_size)
            uc_res = _raw_tensor_2d(uc_addr_residual, BFloat16, self.rows, self.hidden_size)
            uc_w = _raw_tensor_1d(uc_addr_weight, BFloat16, self.hidden_size)
            mc_output = _raw_tensor_1d(mc_addr_output, BFloat16, self.numel * 2)
            uc_sync = _raw_tensor_1d(uc_addr_sync, Int32, self.world_size + 2)
            mc_sync = _raw_tensor_1d(mc_addr_sync, Int32, self.world_size + 2)

            self.kernel(
                mc_in,
                uc_res,
                uc_w,
                mc_output,
                uc_sync,
                mc_sync,
                eps,
            ).launch(
                grid=[self.grid_rows, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )


    class _TileFusedTwoShotBf16Launcher:
        def __init__(
            self,
            rows: int,
            hidden_size: int,
            threads_per_cta: int,
            worker_ctas: int,
            world_size: int,
            rank: int,
        ):
            self.rows = rows
            self.hidden_size = hidden_size
            self.threads_per_cta = threads_per_cta
            self.worker_ctas = worker_ctas
            self.world_size = world_size
            self.rank = rank
            self.vec_chunks = hidden_size // (threads_per_cta * 8)
            self.total_tiles = rows * self.vec_chunks
            self.real_owned_tiles = (
                (self.total_tiles + world_size - 1 - rank) // world_size
                if rank < self.total_tiles
                else 0
            )
            self.numel = rows * hidden_size
            self.phase_sync_size = world_size + 2
            self.phase2_sync_base = self.phase_sync_size

        @cute.kernel
        def kernel(
            self,
            mc_in: cute.Tensor,
            uc_residual: cute.Tensor,
            uc_weight: cute.Tensor,
            mc_output: cute.Tensor,
            uc_output: cute.Tensor,
            mc_partials: cute.Tensor,
            uc_partials: cute.Tensor,
            uc_sync: cute.Tensor,
            mc_sync: cute.Tensor,
            eps: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            worker_idx, _, _ = cute.arch.block_idx()

            if worker_idx == self.worker_ctas:
                if tidx == 0:
                    self._sync_phase(uc_sync, mc_sync, 0)
                    self._sync_phase(uc_sync, mc_sync, self.phase2_sync_base)
            else:
                num_warps = self.threads_per_cta // 32
                smem_ptr = cute.arch.alloc_smem(Float32, num_warps + 1)
                smem = cute.make_tensor(smem_ptr, cute.make_layout((num_warps + 1,)))
                phase1_seen = _ld_global_i32_volatile(uc_sync.iterator + 2 + self.rank)
                phase2_seen = _ld_global_i32_volatile(
                    uc_sync.iterator + self.phase2_sync_base + 2 + self.rank
                )

                owned_idx = worker_idx
                while owned_idx < self.real_owned_tiles:
                    tile_id = owned_idx * self.world_size + self.rank
                    self._compute_h_tile(
                        tidx,
                        tile_id,
                        smem,
                        mc_in,
                        uc_residual,
                        mc_output,
                        mc_partials,
                    )
                    owned_idx += self.worker_ctas

                cute.arch.barrier()
                if tidx == 0:
                    _fence_acq_rel_sys()
                    _atom_global_add_i32(uc_sync.iterator + 0, Int32(1))
                    for peer in cutlass.range_constexpr(self.world_size):
                        while _ld_global_i32_volatile(uc_sync.iterator + 2 + peer) <= phase1_seen:
                            _spin_pause()
                cute.arch.barrier()

                owned_idx = worker_idx
                while owned_idx < self.real_owned_tiles:
                    tile_id = owned_idx * self.world_size + self.rank
                    self._compute_y_tile(
                        tidx,
                        tile_id,
                        uc_weight,
                        mc_output,
                        uc_output,
                        uc_partials,
                        eps,
                    )
                    owned_idx += self.worker_ctas

                cute.arch.barrier()
                if tidx == 0:
                    _fence_acq_rel_sys()
                    _atom_global_add_i32(
                        uc_sync.iterator + self.phase2_sync_base + 0, Int32(1)
                    )
                    for peer in cutlass.range_constexpr(self.world_size):
                        while (
                            _ld_global_i32_volatile(
                                uc_sync.iterator + self.phase2_sync_base + 2 + peer
                            )
                            <= phase2_seen
                        ):
                            _spin_pause()
                cute.arch.barrier()

        @cute.jit
        def _sync_phase(
            self,
            uc_sync: cute.Tensor,
            mc_sync: cute.Tensor,
            base: cutlass.Constexpr,
        ):
            epoch = _atom_global_add_i32(uc_sync.iterator + base + 1, Int32(1)) + Int32(1)
            target = epoch * Int32(self.worker_ctas)
            while _ld_global_i32_volatile(uc_sync.iterator + base + 0) < target:
                _spin_pause()

            _fence_acq_rel_sys()
            _multimem_st_i32(mc_sync.iterator + base + 2 + self.rank, epoch)

        @cute.jit
        def _block_sum(self, value, smem: cute.Tensor):
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            num_warps = self.threads_per_cta // 32

            value = cute.arch.warp_reduction_sum(value)
            if lane_idx == 0:
                smem[warp_idx] = value
            cute.arch.barrier()

            if warp_idx == 0:
                part = Float32(0.0)
                if lane_idx < num_warps:
                    part = smem[lane_idx]
                part = cute.arch.warp_reduction_sum(part)
                if lane_idx == 0:
                    smem[num_warps] = part
            cute.arch.barrier()
            return smem[num_warps]

        @cute.jit
        def _compute_h_tile(
            self,
            tidx,
            tile_id,
            smem: cute.Tensor,
            mc_in: cute.Tensor,
            uc_residual: cute.Tensor,
            mc_output: cute.Tensor,
            mc_partials: cute.Tensor,
        ):
            row = tile_id // self.vec_chunks
            chunk = tile_id - row * self.vec_chunks
            col = chunk * self.threads_per_cta * 8 + tidx * 8
            offset = row * self.hidden_size + col

            r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(
                mc_in.iterator + offset
            )
            b0, b1 = _i32_to_bf16_pair(r0)
            b2, b3 = _i32_to_bf16_pair(r1)
            b4, b5 = _i32_to_bf16_pair(r2)
            b6, b7 = _i32_to_bf16_pair(r3)

            res_r0, res_r1, res_r2, res_r3 = _ld_global_8xbf16(
                uc_residual.iterator + offset
            )
            res0, res1 = _i32_to_bf16_pair(res_r0)
            res2, res3 = _i32_to_bf16_pair(res_r1)
            res4, res5 = _i32_to_bf16_pair(res_r2)
            res6, res7 = _i32_to_bf16_pair(res_r3)

            h0 = Float32(b0) + Float32(res0)
            h1 = Float32(b1) + Float32(res1)
            h2 = Float32(b2) + Float32(res2)
            h3 = Float32(b3) + Float32(res3)
            h4 = Float32(b4) + Float32(res4)
            h5 = Float32(b5) + Float32(res5)
            h6 = Float32(b6) + Float32(res6)
            h7 = Float32(b7) + Float32(res7)

            _multimem_st_8xbf16(
                mc_output.iterator + offset,
                _f32x2_to_bf16x2(h0, h1),
                _f32x2_to_bf16x2(h2, h3),
                _f32x2_to_bf16x2(h4, h5),
                _f32x2_to_bf16x2(h6, h7),
            )

            sum_sq = (
                h0 * h0
                + h1 * h1
                + h2 * h2
                + h3 * h3
                + h4 * h4
                + h5 * h5
                + h6 * h6
                + h7 * h7
            )
            tile_sum = self._block_sum(sum_sq, smem)
            if tidx == 0:
                _multimem_st_f32(mc_partials.iterator + row * self.vec_chunks + chunk, tile_sum)

        @cute.jit
        def _compute_y_tile(
            self,
            tidx,
            tile_id,
            uc_weight: cute.Tensor,
            mc_output: cute.Tensor,
            uc_output: cute.Tensor,
            uc_partials: cute.Tensor,
            eps: Float32,
        ):
            row = tile_id // self.vec_chunks
            chunk = tile_id - row * self.vec_chunks
            col = chunk * self.threads_per_cta * 8 + tidx * 8
            offset = row * self.hidden_size + col

            row_sum = Float32(0.0)
            for c in cutlass.range_constexpr(self.vec_chunks):
                row_sum += _ld_global_f32(uc_partials.iterator + row * self.vec_chunks + c)
            inv_rms = cute.math.rsqrt(row_sum / Float32(self.hidden_size) + eps, fastmath=True)

            h_r0, h_r1, h_r2, h_r3 = _ld_global_8xbf16(uc_output.iterator + offset)
            h0, h1 = _i32_to_bf16_pair(h_r0)
            h2, h3 = _i32_to_bf16_pair(h_r1)
            h4, h5 = _i32_to_bf16_pair(h_r2)
            h6, h7 = _i32_to_bf16_pair(h_r3)

            w_r0, w_r1, w_r2, w_r3 = _ld_global_8xbf16(uc_weight.iterator + col)
            w0, w1 = _i32_to_bf16_pair(w_r0)
            w2, w3 = _i32_to_bf16_pair(w_r1)
            w4, w5 = _i32_to_bf16_pair(w_r2)
            w6, w7 = _i32_to_bf16_pair(w_r3)

            y0 = Float32(h0) * inv_rms * Float32(w0)
            y1 = Float32(h1) * inv_rms * Float32(w1)
            y2 = Float32(h2) * inv_rms * Float32(w2)
            y3 = Float32(h3) * inv_rms * Float32(w3)
            y4 = Float32(h4) * inv_rms * Float32(w4)
            y5 = Float32(h5) * inv_rms * Float32(w5)
            y6 = Float32(h6) * inv_rms * Float32(w6)
            y7 = Float32(h7) * inv_rms * Float32(w7)

            _multimem_st_8xbf16(
                mc_output.iterator + self.numel + offset,
                _f32x2_to_bf16x2(y0, y1),
                _f32x2_to_bf16x2(y2, y3),
                _f32x2_to_bf16x2(y4, y5),
                _f32x2_to_bf16x2(y6, y7),
            )

        @cute.jit
        def __call__(
            self,
            mc_addr_in: Int64,
            uc_addr_residual: Int64,
            uc_addr_weight: Int64,
            mc_addr_output: Int64,
            uc_addr_output: Int64,
            mc_addr_partials: Int64,
            uc_addr_partials: Int64,
            uc_addr_sync: Int64,
            mc_addr_sync: Int64,
            eps: Float32,
            stream: cuda.CUstream,
        ):
            mc_in = _raw_tensor_2d(mc_addr_in, BFloat16, self.rows, self.hidden_size)
            uc_res = _raw_tensor_2d(uc_addr_residual, BFloat16, self.rows, self.hidden_size)
            uc_w = _raw_tensor_1d(uc_addr_weight, BFloat16, self.hidden_size)
            mc_output = _raw_tensor_1d(mc_addr_output, BFloat16, self.numel * 2)
            uc_output = _raw_tensor_1d(uc_addr_output, BFloat16, self.numel * 2)
            mc_partials = _raw_tensor_1d(mc_addr_partials, Float32, self.rows * self.vec_chunks)
            uc_partials = _raw_tensor_1d(uc_addr_partials, Float32, self.rows * self.vec_chunks)
            uc_sync = _raw_tensor_1d(uc_addr_sync, Int32, self.phase_sync_size * 2)
            mc_sync = _raw_tensor_1d(mc_addr_sync, Int32, self.phase_sync_size * 2)

            self.kernel(
                mc_in,
                uc_res,
                uc_w,
                mc_output,
                uc_output,
                mc_partials,
                uc_partials,
                uc_sync,
                mc_sync,
                eps,
            ).launch(
                grid=[self.worker_ctas + 1, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )


def _make_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
) -> _Runtime:
    _require_cute()
    rows, hidden_size = input_2d.shape
    symm_input = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    mc_addr = symm_input.multicast_ptr
    if mc_addr is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    h_out = torch.empty_like(input_2d)
    y_out = torch.empty_like(input_2d)
    compile_args = (
        Int64(mc_addr),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        make_fake_stream(),
    )
    launcher = _FusedArRmsnormBf16Launcher(
        rows,
        hidden_size,
        threads_per_cta,
    )
    launch = cute.compile(launcher, *compile_args)
    return _Runtime(
        symm_input=symm_input,
        launch=launch,
        threads_per_cta=threads_per_cta,
    )


def _get_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
) -> _Runtime:
    key = (
        torch.cuda.current_device(),
        input_2d.shape[0],
        input_2d.shape[1],
        input_2d.dtype,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None or runtime.threads_per_cta != threads_per_cta:
        runtime = _make_runtime(
            input_2d,
            residual_2d,
            weight,
            eps,
            threads_per_cta,
        )
        _RUNTIME_CACHE[key] = runtime
    return runtime


def _make_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduce_threads_per_cta: int,
    rmsnorm_threads_per_cta: int,
    tile_ownership: bool,
) -> _TwoShotRuntime:
    _require_cute()
    rows, hidden_size = input_2d.shape
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    symm_input = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    symm_reduced = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    mc_addr_in = symm_input.multicast_ptr
    mc_addr_reduced = symm_reduced.multicast_ptr
    if mc_addr_in is None or mc_addr_reduced is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    h_out = torch.empty_like(input_2d)
    y_out = torch.empty_like(input_2d)

    reduce_broadcast_launcher = _TwoShotReduceBroadcastBf16Launcher(
        rows,
        hidden_size,
        reduce_threads_per_cta,
        world_size,
        rank,
        tile_ownership,
    )
    reduce_broadcast_launch = cute.compile(
        reduce_broadcast_launcher,
        Int64(mc_addr_in),
        Int64(mc_addr_reduced),
        make_fake_stream(),
    )

    rmsnorm_launcher = _TwoShotRmsnormBf16Launcher(
        rows,
        hidden_size,
        rmsnorm_threads_per_cta,
    )
    rmsnorm_launch = cute.compile(
        rmsnorm_launcher,
        Int64(symm_reduced.tensor.data_ptr()),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        make_fake_stream(),
    )

    return _TwoShotRuntime(
        symm_input=symm_input,
        symm_reduced=symm_reduced,
        reduce_broadcast_launch=reduce_broadcast_launch,
        rmsnorm_launch=rmsnorm_launch,
        reduce_threads_per_cta=reduce_threads_per_cta,
        rmsnorm_threads_per_cta=rmsnorm_threads_per_cta,
    )


def _get_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    reduce_threads_per_cta: int,
    rmsnorm_threads_per_cta: int,
    tile_ownership: bool,
) -> _TwoShotRuntime:
    key = (
        torch.cuda.current_device(),
        input_2d.shape[0],
        input_2d.shape[1],
        input_2d.dtype,
        dist.get_world_size(),
        dist.get_rank(),
        reduce_threads_per_cta,
        rmsnorm_threads_per_cta,
        tile_ownership,
    )
    runtime = _TWOSHOT_RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = _make_twoshot_runtime(
            input_2d,
            residual_2d,
            weight,
            eps,
            reduce_threads_per_cta,
            rmsnorm_threads_per_cta,
            tile_ownership,
        )
        _TWOSHOT_RUNTIME_CACHE[key] = runtime
    return runtime


def _make_fused_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
    device_sync: bool,
) -> _FusedTwoShotRuntime:
    _require_cute()
    rows, hidden_size = input_2d.shape
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    symm_input = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    symm_output = _alloc_symm_tensor(input_2d.numel() * 2, input_2d.dtype)
    symm_sync = _alloc_symm_tensor(world_size + 2, torch.int32)
    mc_addr_in = symm_input.multicast_ptr
    mc_addr_output = symm_output.multicast_ptr
    mc_addr_sync = symm_sync.multicast_ptr
    if mc_addr_in is None or mc_addr_output is None or mc_addr_sync is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    launcher = _FusedTwoShotBf16Launcher(
        rows,
        hidden_size,
        threads_per_cta,
        world_size,
        rank,
        device_sync,
    )
    launch = cute.compile(
        launcher,
        Int64(mc_addr_in),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(mc_addr_output),
        Int64(symm_sync.tensor.data_ptr()),
        Int64(mc_addr_sync),
        Float32(eps),
        make_fake_stream(),
    )

    return _FusedTwoShotRuntime(
        symm_input=symm_input,
        symm_output=symm_output,
        symm_sync=symm_sync,
        launch=launch,
        threads_per_cta=threads_per_cta,
    )


def _get_fused_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
    device_sync: bool,
) -> _FusedTwoShotRuntime:
    key = (
        torch.cuda.current_device(),
        input_2d.shape[0],
        input_2d.shape[1],
        input_2d.dtype,
        dist.get_world_size(),
        dist.get_rank(),
        threads_per_cta,
        device_sync,
    )
    runtime = _FUSED_TWOSHOT_RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = _make_fused_twoshot_runtime(
            input_2d,
            residual_2d,
            weight,
            eps,
            threads_per_cta,
            device_sync,
        )
        _FUSED_TWOSHOT_RUNTIME_CACHE[key] = runtime
    return runtime


def _make_tile_fused_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
    worker_ctas: int,
) -> _TileFusedTwoShotRuntime:
    _require_cute()
    rows, hidden_size = input_2d.shape
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    vec_chunks = hidden_size // (threads_per_cta * 8)

    symm_input = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    symm_output = _alloc_symm_tensor(input_2d.numel() * 2, input_2d.dtype)
    symm_partials = _alloc_symm_tensor(rows * vec_chunks, torch.float32)
    symm_sync = _alloc_symm_tensor((world_size + 2) * 2, torch.int32)

    mc_addr_in = symm_input.multicast_ptr
    mc_addr_output = symm_output.multicast_ptr
    mc_addr_partials = symm_partials.multicast_ptr
    mc_addr_sync = symm_sync.multicast_ptr
    if (
        mc_addr_in is None
        or mc_addr_output is None
        or mc_addr_partials is None
        or mc_addr_sync is None
    ):
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    launcher = _TileFusedTwoShotBf16Launcher(
        rows,
        hidden_size,
        threads_per_cta,
        worker_ctas,
        world_size,
        rank,
    )
    launch = cute.compile(
        launcher,
        Int64(mc_addr_in),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(mc_addr_output),
        Int64(symm_output.tensor.data_ptr()),
        Int64(mc_addr_partials),
        Int64(symm_partials.tensor.data_ptr()),
        Int64(symm_sync.tensor.data_ptr()),
        Int64(mc_addr_sync),
        Float32(eps),
        make_fake_stream(),
    )

    return _TileFusedTwoShotRuntime(
        symm_input=symm_input,
        symm_output=symm_output,
        symm_partials=symm_partials,
        symm_sync=symm_sync,
        launch=launch,
        threads_per_cta=threads_per_cta,
        worker_ctas=worker_ctas,
    )


def _get_tile_fused_twoshot_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
    worker_ctas: int,
) -> _TileFusedTwoShotRuntime:
    key = (
        torch.cuda.current_device(),
        input_2d.shape[0],
        input_2d.shape[1],
        input_2d.dtype,
        dist.get_world_size(),
        dist.get_rank(),
        threads_per_cta,
        worker_ctas,
    )
    runtime = _TILE_FUSED_TWOSHOT_RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = _make_tile_fused_twoshot_runtime(
            input_2d,
            residual_2d,
            weight,
            eps,
            threads_per_cta,
            worker_ctas,
        )
        _TILE_FUSED_TWOSHOT_RUNTIME_CACHE[key] = runtime
    return runtime


def _launch_runtime(
    runtime: _Runtime,
    mc_addr: int,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    h_out: torch.Tensor,
    y_out: torch.Tensor,
    eps: float,
) -> None:
    runtime.launch(
        Int64(mc_addr),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        _current_cu_stream(),
    )


def _launch_twoshot_reduce_broadcast(
    runtime: _TwoShotRuntime,
    mc_addr_in: int,
    mc_addr_reduced: int,
) -> None:
    runtime.reduce_broadcast_launch(
        Int64(mc_addr_in),
        Int64(mc_addr_reduced),
        _current_cu_stream(),
    )


def _launch_twoshot_rmsnorm(
    runtime: _TwoShotRuntime,
    reduced_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    h_out: torch.Tensor,
    y_out: torch.Tensor,
    eps: float,
) -> None:
    runtime.rmsnorm_launch(
        Int64(reduced_2d.data_ptr()),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        _current_cu_stream(),
    )


def _launch_fused_twoshot(
    runtime: _FusedTwoShotRuntime,
    mc_addr_in: int,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    mc_addr_output: int,
    sync: torch.Tensor,
    mc_addr_sync: int,
    eps: float,
) -> None:
    runtime.launch(
        Int64(mc_addr_in),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(mc_addr_output),
        Int64(sync.data_ptr()),
        Int64(mc_addr_sync),
        Float32(eps),
        _current_cu_stream(),
    )


def _launch_tile_fused_twoshot(
    runtime: _TileFusedTwoShotRuntime,
    mc_addr_in: int,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    mc_addr_output: int,
    output: torch.Tensor,
    mc_addr_partials: int,
    partials: torch.Tensor,
    sync: torch.Tensor,
    mc_addr_sync: int,
    eps: float,
) -> None:
    runtime.launch(
        Int64(mc_addr_in),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(mc_addr_output),
        Int64(output.data_ptr()),
        Int64(mc_addr_partials),
        Int64(partials.data_ptr()),
        Int64(sync.data_ptr()),
        Int64(mc_addr_sync),
        Float32(eps),
        _current_cu_stream(),
    )


class FusedAllreduceResidualRmsnormBf16Context:
    """Fixed-shape context for the experimental CuTe POC.

    The safe public wrapper owns synchronization on every call. This context
    exposes the lower-level contract needed to measure or integrate the kernel
    without repeatedly allocating outputs or staging inputs.

    ``sync_mode="internal"`` performs an entry and exit symmetric-memory barrier
    around each ``forward`` call. ``sync_mode="caller"`` does not; the caller
    must guarantee that all ranks have made their symmetric input writes visible
    before launch and that no rank reuses the input region until all peer reads
    are complete.

    Returned output tensors are owned by the context and are overwritten by the
    next ``forward`` call.
    """

    def __init__(
        self,
        reference_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        sync_mode: str = _SYNC_MODE_INTERNAL,
    ):
        _require_cute()
        sync_mode = _validate_sync_mode(sync_mode)
        hidden_size = _check_common_inputs(reference_input, reference_input, weight)

        self.shape = reference_input.shape
        self.hidden_size = hidden_size
        self.eps = eps
        self.sync_mode = sync_mode
        self.weight = weight.contiguous()

        reference_2d = reference_input.contiguous().view(-1, hidden_size)
        self.input = allocate_symmetric_tensor_like(reference_2d).view_as(reference_2d)
        self.input_2d = self.input.view(-1, hidden_size)
        self.h_out = torch.empty_like(self.input_2d)
        self.y_out = torch.empty_like(self.input_2d)

        self.threads_per_cta = _valid_threads_per_cta(hidden_size)

        _enable_symmetric_memory()
        cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
        self.runtime = _get_runtime(
            self.input_2d,
            reference_2d,
            self.weight,
            eps,
            self.threads_per_cta,
        )

        symm_input = _lookup_symmetric_input(self.input_2d)
        if symm_input is None:
            raise RuntimeError("failed to find registered symmetric input")
        self._symm_input = symm_input
        mc_addr = symm_input.multicast_ptr
        if mc_addr is None:
            raise NotImplementedError("symmetric memory multicast pointer is unavailable")
        self._mc_addr = mc_addr

    @property
    def symmetric_input(self) -> torch.Tensor:
        """Window-backed input tensor that a caller may produce into directly."""
        return self.input.view(self.shape)

    def copy_input_(self, input: torch.Tensor) -> torch.Tensor:
        """Stage ``input`` into this context's symmetric input buffer."""
        if input.shape != self.shape:
            raise NotImplementedError("input shape must match the context shape")
        if input.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 input is supported")
        self.input_2d.copy_(input.contiguous().view(-1, self.hidden_size))
        return self.symmetric_input

    def input_ready(self) -> None:
        """Collectively publish prior writes to the symmetric input window."""
        self._symm_input.handle.barrier()

    def input_consumed(self) -> None:
        """Collectively wait until peer reads of the input window are complete."""
        self._symm_input.handle.barrier()

    def _prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape != self.shape:
            raise NotImplementedError("residual shape must match the context shape")
        if residual.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 residual is supported")
        return residual.contiguous().view(-1, self.hidden_size)

    def prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Validate and reshape a residual tensor for repeated context launches."""
        return self._prepare_residual(residual)

    def forward_prepared_assuming_synchronized(
        self,
        residual_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run with a prepared residual and no barriers.

        This is the lowest-overhead context entry point. The caller owns the
        synchronization contract and must pass a residual prepared by
        ``prepare_residual`` for this context's fixed shape.
        """
        _launch_runtime(
            self.runtime,
            self._mc_addr,
            residual_2d,
            self.weight,
            self.h_out,
            self.y_out,
            self.eps,
        )
        return self.y_out.view(self.shape), self.h_out.view(self.shape)

    def make_prepared_forward(
        self,
        residual: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        """Create a low-overhead callable for repeated caller-synchronized runs."""
        residual_2d = self.prepare_residual(residual)
        mc_addr = Int64(self._mc_addr)
        residual_addr = Int64(residual_2d.data_ptr())
        weight_addr = Int64(self.weight.data_ptr())
        h_out_addr = Int64(self.h_out.data_ptr())
        y_out_addr = Int64(self.y_out.data_ptr())
        eps = Float32(self.eps)
        y_view = self.y_out.view(self.shape)
        h_view = self.h_out.view(self.shape)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            self.runtime.launch(
                mc_addr,
                residual_addr,
                weight_addr,
                h_out_addr,
                y_out_addr,
                eps,
                _current_cu_stream(),
            )
            return y_view, h_view

        return run

    def forward(
        self,
        residual: torch.Tensor,
        input: torch.Tensor | None = None,
        *,
        sync: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input is not None:
            self.copy_input_(input)

        residual_2d = self._prepare_residual(residual)
        do_sync = self.sync_mode == _SYNC_MODE_INTERNAL if sync is None else sync
        if do_sync:
            self.input_ready()
        outputs = self.forward_prepared_assuming_synchronized(residual_2d)
        if do_sync:
            self.input_consumed()
        return outputs

    def forward_assuming_synchronized(
        self,
        residual: torch.Tensor,
        input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run without barriers; caller owns input visibility and lifetime."""
        return self.forward(residual, input, sync=False)

    __call__ = forward


class FusedAllreduceResidualRmsnormBf16TwoShotContext:
    """Fixed-shape two-shot variant of the CuTe POC.

    Stage 1 reduce-scatters token ownership with ``multimem.ld_reduce`` and
    broadcasts reduced chunks with ``multimem.st``. Stage 2 runs residual +
    RMSNorm from the broadcast symmetric buffer. A symmetric-memory barrier is
    required between stages unless an integration provides an equivalent
    cross-rank producer/consumer protocol.
    """

    def __init__(
        self,
        reference_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        sync_mode: str = _SYNC_MODE_INTERNAL,
        device_sync: bool = False,
    ):
        _require_cute()
        sync_mode = _validate_sync_mode(sync_mode)
        hidden_size = _check_common_inputs(reference_input, reference_input, weight)

        self.shape = reference_input.shape
        self.hidden_size = hidden_size
        self.eps = eps
        self.sync_mode = sync_mode
        self.weight = weight.contiguous()

        reference_2d = reference_input.contiguous().view(-1, hidden_size)
        self.input = allocate_symmetric_tensor_like(reference_2d).view_as(reference_2d)
        self.reduced = allocate_symmetric_tensor_like(reference_2d).view_as(reference_2d)
        self.input_2d = self.input.view(-1, hidden_size)
        self.reduced_2d = self.reduced.view(-1, hidden_size)
        self.h_out = torch.empty_like(self.input_2d)
        self.y_out = torch.empty_like(self.input_2d)

        world_size = dist.get_world_size()
        self.reduce_threads_per_cta = _valid_twoshot_reduce_threads_per_cta(
            reference_2d.shape[0],
            hidden_size,
            world_size,
        )
        self.rmsnorm_threads_per_cta = _valid_twoshot_rmsnorm_threads_per_cta(
            reference_2d.shape[0],
            hidden_size,
        )
        self.threads_per_cta = self.rmsnorm_threads_per_cta
        self.twoshot_tile_ownership = _use_twoshot_tile_ownership(
            reference_2d.shape[0], world_size
        )

        _enable_symmetric_memory()
        cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
        self.runtime = _get_twoshot_runtime(
            self.input_2d,
            reference_2d,
            self.weight,
            eps,
            self.reduce_threads_per_cta,
            self.rmsnorm_threads_per_cta,
            self.twoshot_tile_ownership,
        )

        symm_input = _lookup_symmetric_input(self.input_2d)
        symm_reduced = _lookup_symmetric_input(self.reduced_2d)
        if symm_input is None or symm_reduced is None:
            raise RuntimeError("failed to find registered symmetric buffers")

        self._symm_input = symm_input
        self._symm_reduced = symm_reduced
        mc_addr_in = symm_input.multicast_ptr
        mc_addr_reduced = symm_reduced.multicast_ptr
        if mc_addr_in is None or mc_addr_reduced is None:
            raise NotImplementedError("symmetric memory multicast pointer is unavailable")
        self._mc_addr_in = mc_addr_in
        self._mc_addr_reduced = mc_addr_reduced

    @property
    def symmetric_input(self) -> torch.Tensor:
        return self.input.view(self.shape)

    def copy_input_(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape != self.shape:
            raise NotImplementedError("input shape must match the context shape")
        if input.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 input is supported")
        self.input_2d.copy_(input.contiguous().view(-1, self.hidden_size))
        return self.symmetric_input

    def input_ready(self) -> None:
        self._symm_input.handle.barrier()

    def reduced_ready(self) -> None:
        self._symm_reduced.handle.barrier()

    def input_consumed(self) -> None:
        self._symm_input.handle.barrier()

    def _prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape != self.shape:
            raise NotImplementedError("residual shape must match the context shape")
        if residual.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 residual is supported")
        return residual.contiguous().view(-1, self.hidden_size)

    def prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self._prepare_residual(residual)

    def launch_reduce_broadcast(self) -> None:
        _launch_twoshot_reduce_broadcast(
            self.runtime,
            self._mc_addr_in,
            self._mc_addr_reduced,
        )

    def launch_rmsnorm(self, residual_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _launch_twoshot_rmsnorm(
            self.runtime,
            self.reduced_2d,
            residual_2d,
            self.weight,
            self.h_out,
            self.y_out,
            self.eps,
        )
        return self.y_out.view(self.shape), self.h_out.view(self.shape)

    def make_prepared_forward(
        self,
        residual: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        residual_2d = self.prepare_residual(residual)
        y_view = self.y_out.view(self.shape)
        h_view = self.h_out.view(self.shape)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            self.launch_reduce_broadcast()
            self.reduced_ready()
            self.launch_rmsnorm(residual_2d)
            return y_view, h_view

        return run

    def forward(
        self,
        residual: torch.Tensor,
        input: torch.Tensor | None = None,
        *,
        sync: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input is not None:
            self.copy_input_(input)

        residual_2d = self._prepare_residual(residual)
        do_sync = self.sync_mode == _SYNC_MODE_INTERNAL if sync is None else sync
        if do_sync:
            self.input_ready()
        self.launch_reduce_broadcast()
        self.reduced_ready()
        outputs = self.launch_rmsnorm(residual_2d)
        if do_sync:
            self.input_consumed()
        return outputs

    __call__ = forward


class FusedAllreduceResidualRmsnormBf16FusedTwoShotContext:
    """Single-kernel row-owner variant of the two-shot CuTe POC.

    One rank owns each row, performs the full ``multimem.ld_reduce`` +
    residual + RMSNorm computation for that row, and multicast-stores final
    ``h`` and ``y`` into a symmetric output buffer. This keeps the operation in
    one CUDA kernel, but assumes residual/weight are replicated across ranks.
    """

    def __init__(
        self,
        reference_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        sync_mode: str = _SYNC_MODE_INTERNAL,
        device_sync: bool = False,
    ):
        _require_cute()
        sync_mode = _validate_sync_mode(sync_mode)
        hidden_size = _check_common_inputs(reference_input, reference_input, weight)

        self.shape = reference_input.shape
        self.hidden_size = hidden_size
        self.eps = eps
        self.sync_mode = sync_mode
        self.device_sync = device_sync
        self.weight = weight.contiguous()

        reference_2d = reference_input.contiguous().view(-1, hidden_size)
        world_size = dist.get_world_size()
        output_reference = torch.empty(
            (2, reference_2d.numel()),
            dtype=reference_2d.dtype,
            device=reference_2d.device,
        )
        self.input = allocate_symmetric_tensor_like(reference_2d).view_as(reference_2d)
        self.output = allocate_symmetric_tensor_like(output_reference).view(2, reference_2d.numel())
        self.input_2d = self.input.view(-1, hidden_size)
        self.h_out = self.output[0].view_as(self.input_2d)
        self.y_out = self.output[1].view_as(self.input_2d)
        if device_sync:
            sync_reference = torch.empty(
                (world_size + 2,), dtype=torch.int32, device=reference_2d.device
            )
            self.sync = allocate_symmetric_tensor_like(sync_reference).view(-1)
            self.sync.zero_()
            torch.cuda.synchronize()
        else:
            self.sync = None

        self.threads_per_cta = _valid_fused_twoshot_threads_per_cta(
            reference_2d.shape[0],
            hidden_size,
        )

        _enable_symmetric_memory()
        cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
        self.runtime = _get_fused_twoshot_runtime(
            self.input_2d,
            reference_2d,
            self.weight,
            eps,
            self.threads_per_cta,
            device_sync,
        )

        symm_input = _lookup_symmetric_input(self.input_2d)
        symm_output = _lookup_symmetric_input(self.output.reshape(-1))
        symm_sync = (
            _lookup_symmetric_input(self.sync)
            if self.sync is not None
            else self.runtime.symm_sync
        )
        if symm_input is None or symm_output is None or symm_sync is None:
            raise RuntimeError("failed to find registered symmetric buffers")

        self._symm_input = symm_input
        self._symm_output = symm_output
        self._symm_sync = symm_sync
        mc_addr_in = symm_input.multicast_ptr
        mc_addr_output = symm_output.multicast_ptr
        mc_addr_sync = symm_sync.multicast_ptr
        if mc_addr_in is None or mc_addr_output is None or mc_addr_sync is None:
            raise NotImplementedError("symmetric memory multicast pointer is unavailable")
        self._mc_addr_in = mc_addr_in
        self._mc_addr_output = mc_addr_output
        self._mc_addr_sync = mc_addr_sync
        if self.sync is None:
            self.sync = symm_sync.tensor
        self.world_size = world_size

    @property
    def symmetric_input(self) -> torch.Tensor:
        return self.input.view(self.shape)

    def copy_input_(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape != self.shape:
            raise NotImplementedError("input shape must match the context shape")
        if input.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 input is supported")
        self.input_2d.copy_(input.contiguous().view(-1, self.hidden_size))
        return self.symmetric_input

    def input_ready(self) -> None:
        self._symm_input.handle.barrier()

    def output_ready(self) -> None:
        self._symm_output.handle.barrier()

    def _prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape != self.shape:
            raise NotImplementedError("residual shape must match the context shape")
        if residual.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 residual is supported")
        return residual.contiguous().view(-1, self.hidden_size)

    def prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self._prepare_residual(residual)

    def launch_fused(self, residual_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _launch_fused_twoshot(
            self.runtime,
            self._mc_addr_in,
            residual_2d,
            self.weight,
            self._mc_addr_output,
            self.sync,
            self._mc_addr_sync,
            self.eps,
        )
        return self.y_out.view(self.shape), self.h_out.view(self.shape)

    def make_prepared_forward(
        self,
        residual: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        residual_2d = self.prepare_residual(residual)
        y_view = self.y_out.view(self.shape)
        h_view = self.h_out.view(self.shape)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            self.launch_fused(residual_2d)
            if not self.device_sync:
                self.output_ready()
            return y_view, h_view

        return run

    def forward(
        self,
        residual: torch.Tensor,
        input: torch.Tensor | None = None,
        *,
        sync: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input is not None:
            self.copy_input_(input)

        residual_2d = self._prepare_residual(residual)
        do_sync = self.sync_mode == _SYNC_MODE_INTERNAL if sync is None else sync
        if do_sync:
            self.input_ready()
        outputs = self.launch_fused(residual_2d)
        if not self.device_sync:
            self.output_ready()
        return outputs

    __call__ = forward


class FusedAllreduceResidualRmsnormBf16FusedTwoShotDeviceSyncContext(
    FusedAllreduceResidualRmsnormBf16FusedTwoShotContext
):
    """Experimental true-fused two-shot context with an in-kernel completion protocol."""

    def __init__(
        self,
        reference_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        sync_mode: str = _SYNC_MODE_INTERNAL,
    ):
        super().__init__(
            reference_input,
            weight,
            eps,
            sync_mode=sync_mode,
            device_sync=True,
        )


class FusedAllreduceResidualRmsnormBf16TileFusedTwoShotDeviceSyncContext:
    """Tile-owned single-kernel two-shot experiment with in-kernel completion."""

    def __init__(
        self,
        reference_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        sync_mode: str = _SYNC_MODE_INTERNAL,
    ):
        _require_cute()
        sync_mode = _validate_sync_mode(sync_mode)
        hidden_size = _check_common_inputs(reference_input, reference_input, weight)

        self.shape = reference_input.shape
        self.hidden_size = hidden_size
        self.eps = eps
        self.sync_mode = sync_mode
        self.weight = weight.contiguous()

        reference_2d = reference_input.contiguous().view(-1, hidden_size)
        output_reference = torch.empty(
            (2, reference_2d.numel()),
            dtype=reference_2d.dtype,
            device=reference_2d.device,
        )
        self.input = allocate_symmetric_tensor_like(reference_2d).view_as(reference_2d)
        self.output = allocate_symmetric_tensor_like(output_reference).view(
            2, reference_2d.numel()
        )
        self.input_2d = self.input.view(-1, hidden_size)
        self.h_out = self.output[0].view_as(self.input_2d)
        self.y_out = self.output[1].view_as(self.input_2d)

        world_size = dist.get_world_size()
        self.threads_per_cta = _valid_tile_fused_twoshot_threads_per_cta(hidden_size)
        vec_chunks = hidden_size // (self.threads_per_cta * 8)
        total_tiles = reference_2d.shape[0] * vec_chunks
        rank = dist.get_rank()
        owned_tiles = (
            (total_tiles + world_size - 1 - rank) // world_size
            if rank < total_tiles
            else 0
        )
        self.worker_ctas = _tile_fused_twoshot_worker_ctas(owned_tiles)

        sync_reference = torch.empty(
            ((world_size + 2) * 2,), dtype=torch.int32, device=reference_2d.device
        )
        partials_reference = torch.empty(
            (reference_2d.shape[0] * vec_chunks,),
            dtype=torch.float32,
            device=reference_2d.device,
        )
        self.sync = allocate_symmetric_tensor_like(sync_reference).view(-1)
        self.partials = allocate_symmetric_tensor_like(partials_reference).view(-1)
        self.sync.zero_()
        torch.cuda.synchronize()

        _enable_symmetric_memory()
        cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
        self.runtime = _get_tile_fused_twoshot_runtime(
            self.input_2d,
            reference_2d,
            self.weight,
            eps,
            self.threads_per_cta,
            self.worker_ctas,
        )

        symm_input = _lookup_symmetric_input(self.input_2d)
        symm_output = _lookup_symmetric_input(self.output.reshape(-1))
        symm_partials = _lookup_symmetric_input(self.partials)
        symm_sync = _lookup_symmetric_input(self.sync)
        if (
            symm_input is None
            or symm_output is None
            or symm_partials is None
            or symm_sync is None
        ):
            raise RuntimeError("failed to find registered symmetric buffers")

        self._symm_input = symm_input
        self._symm_output = symm_output
        self._symm_partials = symm_partials
        self._symm_sync = symm_sync
        mc_addr_in = symm_input.multicast_ptr
        mc_addr_output = symm_output.multicast_ptr
        mc_addr_partials = symm_partials.multicast_ptr
        mc_addr_sync = symm_sync.multicast_ptr
        if (
            mc_addr_in is None
            or mc_addr_output is None
            or mc_addr_partials is None
            or mc_addr_sync is None
        ):
            raise NotImplementedError("symmetric memory multicast pointer is unavailable")
        self._mc_addr_in = mc_addr_in
        self._mc_addr_output = mc_addr_output
        self._mc_addr_partials = mc_addr_partials
        self._mc_addr_sync = mc_addr_sync

    @property
    def symmetric_input(self) -> torch.Tensor:
        return self.input.view(self.shape)

    def copy_input_(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape != self.shape:
            raise NotImplementedError("input shape must match the context shape")
        if input.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 input is supported")
        self.input_2d.copy_(input.contiguous().view(-1, self.hidden_size))
        return self.symmetric_input

    def input_ready(self) -> None:
        self._symm_input.handle.barrier()

    def _prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape != self.shape:
            raise NotImplementedError("residual shape must match the context shape")
        if residual.dtype != torch.bfloat16:
            raise NotImplementedError("only bf16 residual is supported")
        return residual.contiguous().view(-1, self.hidden_size)

    def prepare_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self._prepare_residual(residual)

    def launch_fused(self, residual_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _launch_tile_fused_twoshot(
            self.runtime,
            self._mc_addr_in,
            residual_2d,
            self.weight,
            self._mc_addr_output,
            self.output,
            self._mc_addr_partials,
            self.partials,
            self.sync,
            self._mc_addr_sync,
            self.eps,
        )
        return self.y_out.view(self.shape), self.h_out.view(self.shape)

    def make_prepared_forward(
        self,
        residual: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        residual_2d = self.prepare_residual(residual)
        y_view = self.y_out.view(self.shape)
        h_view = self.h_out.view(self.shape)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            self.launch_fused(residual_2d)
            return y_view, h_view

        return run

    def forward(
        self,
        residual: torch.Tensor,
        input: torch.Tensor | None = None,
        *,
        sync: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input is not None:
            self.copy_input_(input)

        residual_2d = self._prepare_residual(residual)
        do_sync = self.sync_mode == _SYNC_MODE_INTERNAL if sync is None else sync
        if do_sync:
            self.input_ready()
        return self.launch_fused(residual_2d)

    __call__ = forward


def fused_allreduce_residual_rmsnorm_bf16(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the eager-only CuTe POC.

    Raises:
        NotImplementedError: If this call is outside the narrow supported POC
            envelope.
    """
    _require_cute()
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("CuTe allreduce RMSNorm POC is not CUDA-graph safe")
    hidden_size = _check_common_inputs(input, residual, weight)

    input_2d = input.contiguous().view(-1, hidden_size)
    residual_2d = residual.contiguous().view(-1, hidden_size)
    threads_per_cta = _valid_threads_per_cta(hidden_size)

    _enable_symmetric_memory()
    cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
    runtime = _get_runtime(input_2d, residual_2d, weight, eps, threads_per_cta)
    symm_input = _lookup_symmetric_input(input_2d)
    if symm_input is None:
        symm_input = runtime.symm_input
        symm_input.tensor.copy_(input_2d.reshape(-1))

    mc_addr = symm_input.multicast_ptr
    if mc_addr is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    h_out = torch.empty_like(input_2d)
    y_out = torch.empty_like(input_2d)

    symm_input.handle.barrier()
    _launch_runtime(runtime, mc_addr, residual_2d, weight, h_out, y_out, eps)
    symm_input.handle.barrier()
    return y_out.view_as(input), h_out.view_as(input)


def fused_allreduce_residual_rmsnorm_bf16_twoshot(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cute()
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("CuTe two-shot allreduce RMSNorm POC is not CUDA-graph safe")

    context = FusedAllreduceResidualRmsnormBf16TwoShotContext(input, weight, eps)
    return context.forward(residual, input)


def fused_allreduce_residual_rmsnorm_bf16_fused_twoshot(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cute()
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("CuTe fused two-shot allreduce RMSNorm POC is not CUDA-graph safe")

    context = FusedAllreduceResidualRmsnormBf16FusedTwoShotContext(input, weight, eps)
    return context.forward(residual, input)


def fused_allreduce_residual_rmsnorm_bf16_tile_fused_twoshot(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cute()
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "CuTe tile-fused two-shot allreduce RMSNorm POC is not CUDA-graph safe"
        )

    context = FusedAllreduceResidualRmsnormBf16TileFusedTwoShotDeviceSyncContext(
        input, weight, eps
    )
    return context.forward(residual, input)
