#pragma once
#include <format>
#include <deep_ep/layout/afd/context.hpp>
#include "../../runtime/jit.hpp"

// ### PATCH START: AFD JIT launcher adapted from kernels/ep/dispatch.hpp, 3b25377d
namespace deep_ep::afd {
static void launch_dispatch_send(void* x, void* sf, void* topk_idx, void* topk_weights,
    void* token_mask, void* task_info, void* context, void* control_buffer,
    void* round_ledger, void* workspace, void* jetty_ptrs, void* send_status,
    const AFDContextLayout& config, int num_ub_bytes, int64_t num_timeout_cycles,
    int sf_token_stride, int sf_hidden_stride) {
    const auto kernel=jit->compile("afd_dispatch_send",std::format(R"(
#include <deep_ep/impls/afd/dispatch_send.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
    auto ptr=reinterpret_cast<void*>(&deep_ep::afd::dispatch_impl<{},{},{},{},{},{},{},{},{},{}>);
}}
)",config.num_ranks,config.num_experts,config.num_topk,config.num_vec_cores,
        config.num_hidden_bytes,config.num_sf_packs,config.num_max_tokens_per_rank,
        config.num_tokens_per_tile,num_ub_bytes,num_timeout_cycles));
    jit->launch(kernel,{.num_blocks=int(config.num_vec_cores),.num_ubuf_bytes=num_ub_bytes},
        x,sf,topk_idx,topk_weights,token_mask,task_info,context,control_buffer,
        round_ledger,workspace,jetty_ptrs,send_status,sf_token_stride,sf_hidden_stride);
}
// Source: launch_dispatch_copy_epilogue; compile a selected-piece family, no host counts.
static void launch_dispatch_recv(void* buffer,void* context,void* active,void* execution_state,
    void* workspace,void* recv_x,void* recv_sf,void* recv_src_metadata,void* control_buffer,
    const AFDContextLayout& config,int rank_idx,int q,int q_alloc,int alignment,int tile,
    int num_ub_bytes,int sf_token_stride,int sf_hidden_stride) {
    const auto kernel=jit->compile("afd_dispatch_recv",std::format(R"(
#include <deep_ep/impls/afd/dispatch_recv.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
    auto ptr=reinterpret_cast<void*>(&deep_ep::afd::dispatch_copy_epilogue_impl<{},{},{},{},{},{},{},{},{},{},{},{},{}>);
}}
)",config.num_ranks,config.num_experts,config.num_topk,config.num_vec_cores,
        config.num_hidden_bytes,config.num_sf_packs,config.num_source_lanes,config.num_layers,
        q,q_alloc,alignment,tile,num_ub_bytes));
    jit->launch(kernel,{.num_blocks=int(config.num_vec_cores),.num_ubuf_bytes=num_ub_bytes},
        buffer,context,active,execution_state,workspace,recv_x,recv_sf,recv_src_metadata,
        control_buffer,rank_idx,sf_token_stride,sf_hidden_stride,
        config.payload_bytes,config.control_bytes,config.context_bytes);
}
static std::string compile_dispatch_send_only(const AFDContextLayout& config,int num_ub_bytes,int64_t num_timeout_cycles) {
    return jit->compile_without_load("afd_dispatch_send",std::format(R"(
#include <deep_ep/impls/afd/dispatch_send.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
    auto ptr=reinterpret_cast<void*>(&deep_ep::afd::dispatch_impl<{},{},{},{},{},{},{},{},{},{}>);
}}
)",config.num_ranks,config.num_experts,config.num_topk,config.num_vec_cores,
        config.num_hidden_bytes,config.num_sf_packs,config.num_max_tokens_per_rank,
        config.num_tokens_per_tile,num_ub_bytes,num_timeout_cycles)).string();
}
static std::string compile_dispatch_recv_only(const AFDContextLayout& config,int q,int q_alloc,int alignment,int tile,int num_ub_bytes) {
    return jit->compile_without_load("afd_dispatch_recv",std::format(R"(
#include <deep_ep/impls/afd/dispatch_recv.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
    auto ptr=reinterpret_cast<void*>(&deep_ep::afd::dispatch_copy_epilogue_impl<{},{},{},{},{},{},{},{},{},{},{},{},{}>);
}}
)",config.num_ranks,config.num_experts,config.num_topk,config.num_vec_cores,
        config.num_hidden_bytes,config.num_sf_packs,config.num_source_lanes,config.num_layers,
        q,q_alloc,alignment,tile,num_ub_bytes)).string();
}
} // namespace deep_ep::afd
// ### PATCH END: AFD JIT launcher
