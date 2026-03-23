/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_distance_binary_adc.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

/**
 * @brief Dataset descriptor for binary ADC (Asymmetric Distance Computation).
 *
 * Computes distance between float queries and packed binary vectors.
 *
 * Distance formula (negative dot product for ranking):
 *   dist(q, b) = -dot(q, b) = -sum_d( q[d] * bit(b, d) )
 *
 * Under L2, the full expansion is:
 *   ||q - b||^2 = sum(q^2) - 2*dot(q,b) + popcount(b)
 * Since sum(q^2) is constant for a given query, and popcount(b) is cheap,
 * we use -dot(q,b) as the primary ranking signal. The team_sum in the search
 * kernel will aggregate partial dot products across team lanes.
 *
 * Note on DataT: The spec is instantiated with DataT=float because the CAGRA
 * index is index<float> and search_main dispatches with T=float. The actual
 * dataset storage is uint8_t (packed bits), handled internally.
 *
 * Layout of args_t fields:
 *   extra_ptr1  -> packed binary data pointer (const uint8_t*)
 *   extra_word1 -> packed_dim (bytes per row = ceil(D/8))
 *   smem_ws_ptr -> shared memory workspace (descriptor + query buffer)
 */
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct binary_adc_dataset_descriptor_t
  : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using QUERY_T   = float;
  using base_type::args;
  using typename base_type::args_t;
  using typename base_type::compute_distance_type;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  using typename base_type::setup_workspace_type;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;

  RAFT_INLINE_FUNCTION static constexpr auto packed_data_ptr(args_t& args) noexcept
    -> const uint8_t*&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto packed_data_ptr(const args_t& args) noexcept
    -> const uint8_t* const&
  {
    return (const uint8_t* const&)args.extra_ptr1;
  }

  RAFT_INLINE_FUNCTION static constexpr auto packed_dim(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto packed_dim(const args_t& args) noexcept
    -> const uint32_t&
  {
    return args.extra_word1;
  }

  _RAFT_HOST_DEVICE binary_adc_dataset_descriptor_t(setup_workspace_type* setup_workspace_impl,
                                                     compute_distance_type* compute_distance_impl,
                                                     const uint8_t* packed_data_ptr,
                                                     uint32_t packed_dim,
                                                     IndexT size,
                                                     uint32_t dim)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                raft::Pow2<TeamSize>::Log2,
                get_smem_ws_size_in_bytes(dim))
  {
    binary_adc_dataset_descriptor_t::packed_data_ptr(args) = packed_data_ptr;
    binary_adc_dataset_descriptor_t::packed_dim(args)      = packed_dim;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(binary_adc_dataset_descriptor_t) == alignof(base_type));
  }

 private:
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    /* SMEM workspace layout:
      1. The descriptor itself
      2. Query buffer (dim floats, rounded up to DatasetBlockDim)
      No query_sum needed — we use pure negative dot product for ranking.
    */
    return sizeof(binary_adc_dataset_descriptor_t) +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }
};

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_binary_adc(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  using word_type                 = uint32_t;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;

  auto* r   = reinterpret_cast<DescriptorT*>(smem_ptr);
  auto* buf = reinterpret_cast<QUERY_T*>(r + 1);

  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
    }
    __syncthreads();
  }

  uint32_t dim = r->args.dim;
  auto buf_len = raft::round_up_safe<uint32_t>(dim, kDatasetBlockDim);
  queries_ptr += dim * query_id;

  // Load query into shared memory.
  // No swizzling needed: binary ADC reads are indexed by (byte_idx * 8 + bit_pos),
  // which is sequential. Bank conflicts are minimal for sequential float reads.
  for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
    buf[i] = (i < dim) ? static_cast<QUERY_T>(queries_ptr[i]) : QUERY_T(0);
  }
  __syncthreads();

  return const_cast<const DescriptorT*>(r);
}

/**
 * @brief Compute ADC distance between a float query and a packed binary vector.
 *
 * Returns -dot(q, b) = -sum_d( q[d] * bit(b, d) ).
 *
 * Key performance decisions:
 * - Per-byte global loads via ld.global.cg.u8: safe at any alignment (packed_dim may not
 *   be a multiple of 4, making row starts unaligned for wider loads). Adjacent team threads
 *   read adjacent bytes, which the memory controller coalesces.
 * - Branchless bit extraction: float(bit) * query_val avoids warp divergence.
 *   The multiply by 0.0f or 1.0f is a single FMUL instruction vs a branch
 *   that would serialize half the warp on average with random binary data.
 * - Each thread processes packed bytes strided by TeamSize bytes (TeamSize*8 dimensions).
 *
 * Memory access pattern:
 *   packed_ptr[byte_idx] is read from global memory via ld.global.cg.u8 (cached at L2).
 *   Adjacent team threads read adjacent bytes → coalesced into 32B/128B transactions.
 *   For D=256 (packed_dim=32), with TeamSize=8, each thread processes 4 bytes = 32 bits.
 */
template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_binary_adc(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kTeamMask        = kTeamSize - 1;

  const uint32_t dim       = args.dim;
  const uint32_t p_dim     = DescriptorT::packed_dim(args);
  const auto* packed_ptr   = DescriptorT::packed_data_ptr(args) +
                             static_cast<uint64_t>(p_dim) * dataset_index;
  const uint32_t query_ptr = args.smem_ws_ptr;
  const auto laneId        = threadIdx.x & kTeamMask;

  DISTANCE_T neg_dot = 0;

  // Process packed bytes, one byte at a time per iteration, strided across team lanes.
  // Each byte covers 8 binary dimensions → 8 branchless FMUL+ADD per byte.
  // Stride by TeamSize bytes: threads in a team read adjacent bytes for coalescing.
  for (uint32_t byte_idx = laneId; byte_idx < p_dim; byte_idx += kTeamSize) {
    // Load one byte from global memory. Individual byte loads are safe regardless of
    // alignment. Adjacent threads read adjacent bytes -> coalesced into 32B/128B transactions.
    uint32_t packed_byte;
    asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(packed_byte) : "l"(packed_ptr + byte_idx));

    const uint32_t base_dim = byte_idx * 8;

    // Process 8 bits from this byte. Fully unrolled, branchless.
#pragma unroll
    for (uint32_t bit = 0; bit < 8; bit++) {
      const uint32_t d = base_dim + bit;
      if (d < dim) {
        const float bit_val = static_cast<float>((packed_byte >> bit) & 1u);
        QUERY_T q_val;
        device::lds(q_val, query_ptr + sizeof(QUERY_T) * d);
        neg_dot += static_cast<DISTANCE_T>(bit_val * q_val);
      }
    }
  }

  // Return negative dot product. Lower values = higher similarity = closer neighbors.
  // The search kernel's team_sum aggregates across team lanes.
  return -neg_dot;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1) binary_adc_dataset_descriptor_init_kernel(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
  const uint8_t* packed_data_ptr,
  uint32_t packed_dim,
  IndexT size,
  uint32_t dim)
{
  using desc_type =
    binary_adc_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;
  new (out) desc_type(
    reinterpret_cast<typename base_type::setup_workspace_type*>(
      &setup_workspace_binary_adc<desc_type>),
    reinterpret_cast<typename base_type::compute_distance_type*>(
      &compute_distance_binary_adc<desc_type>),
    packed_data_ptr,
    packed_dim,
    size,
    dim);
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
binary_adc_descriptor_spec<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>::init_(
  const cagra::search_params& params,
  const uint8_t* packed_data_ptr,
  uint32_t packed_dim,
  IndexT size,
  uint32_t dim)
{
  using desc_type =
    binary_adc_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;

  desc_type dd_host{nullptr, nullptr, packed_data_ptr, packed_dim, size, dim};
  return host_type{dd_host,
                   [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
                       rmm::cuda_stream_view stream) {
                     binary_adc_dataset_descriptor_init_kernel<TeamSize,
                                                               DatasetBlockDim,
                                                               DataT,
                                                               IndexT,
                                                               DistanceT>
                       <<<1, 1, 0, stream>>>(dev_ptr, packed_data_ptr, packed_dim, size, dim);
                     RAFT_CUDA_TRY(cudaPeekAtLastError());
                   }};
}

}  // namespace cuvs::neighbors::cagra::detail
