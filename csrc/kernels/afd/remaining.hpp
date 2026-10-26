#pragma once
#include <format>
#include <deep_ep/layout/afd/context.hpp>
#include "../../runtime/jit.hpp"

// ### PATCH START: remaining AFD launchers, native EP DeepJIT conventions
namespace deep_ep::afd {
static void launch_ffn_scheduler(void* buffer,void* context,void* control,void* workspace,
    void* active,void* execution,void* completion,void* psum,void* counts,
    const AFDContextLayout& c,int rank,int q,int qa,int alignment,bool precheck,int ub) {
    const auto kernel=jit->compile("afd_ffn_scheduler",std::format(R"(
#define AFD_KERNEL_FFN_SCHEDULER_TU
#include <deep_ep/impls/afd/ffn_scheduler.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&ffn_scheduler_impl<{},{},{},{},{},{},{},{},{}>);
}}
)",c.num_ranks,c.num_experts,c.num_topk,c.num_source_lanes,c.num_layers,q,qa,alignment,precheck?1:0));
    jit->launch(kernel,{.num_blocks=1,.num_ubuf_bytes=ub},buffer,context,control,workspace,
        active,execution,completion,psum,counts,rank);
}
static void launch_compute_ready(void* active,void* execution,int ub) {
    const auto kernel=jit->compile("afd_compute_ready",R"(
#define AFD_KERNEL_COMPUTE_READY_TU
#include <deep_ep/impls/afd/ffn_scheduler.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {
 auto ptr=reinterpret_cast<void*>(&mark_compute_ready_impl);
}
)");
    jit->launch(kernel,{.num_blocks=1,.num_ubuf_bytes=ub},active,execution);
}
static void launch_combine_send(void* x,void* metadata,void* context,void* active,void* control,
    void* execution,void* workspace,void* jetties,void* completion,
    const AFDContextLayout& c,int rank,int hidden,int qa,int tile,int ub,int64_t timeout) {
    const auto kernel=jit->compile("afd_combine_send",std::format(R"(
#include <deep_ep/impls/afd/combine_send.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&combine_send_impl<{},{},{},{},{},{},{},{},{},{},{},{}>);
}}
)",c.num_src_ranks,c.num_ranks,c.num_experts/c.num_ranks,c.num_topk,hidden,c.num_vec_cores,
        c.num_source_lanes,c.num_layers,qa,tile,ub,timeout));
    jit->launch(kernel,{.num_blocks=int(c.num_vec_cores),.num_ubuf_bytes=ub},x,metadata,context,
        active,control,execution,workspace,jetties,completion,rank);
}
static void launch_combine_recv(void* x,void* idx,void* weights,void* mask,void* buffer,
    void* task,void* context,void* control,void* ledger,void* workspace,void* status,
    const AFDContextLayout& c,int hidden,int chunk,int ub,int64_t timeout,
    void* progress,bool round_step,bool start) {
    const auto kernel=jit->compile("afd_combine_recv",std::format(R"(
#include <deep_ep/impls/afd/combine_recv.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&combine_recv_impl<{},{},{},{},{},{},{},{}>);
}}
)",c.num_experts,c.num_ranks,c.num_topk,hidden,chunk,c.num_vec_cores,timeout,round_step?1:0));
    jit->launch(kernel,{.num_blocks=int(c.num_vec_cores),.num_ubuf_bytes=ub},x,idx,weights,mask,
        buffer,task,context,control,ledger,workspace,status,progress,start);
}
static void launch_queue_meta_probe(void* payload_jetties,void* control_jetties,void* out,int ub) {
    const auto kernel=jit->compile("afd_queue_meta_probe",R"(
#include <deep_ep/impls/afd/queue_meta_probe.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {
 auto ptr=reinterpret_cast<void*>(&queue_meta_probe_impl);
}
)");
    jit->launch(kernel,{.num_blocks=1,.num_ubuf_bytes=ub},payload_jetties,control_jetties,out);
}
static std::string compile_queue_meta_probe_only() {
    return jit->compile_without_load("afd_queue_meta_probe",R"(
#include <deep_ep/impls/afd/queue_meta_probe.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {
 auto ptr=reinterpret_cast<void*>(&queue_meta_probe_impl);
}
)").string();
}
static std::string compile_combine_send_only(const AFDContextLayout& c,int rank,int hidden,int qa,int tile,int ub,int64_t timeout) {
    return jit->compile_without_load("afd_combine_send",std::format(R"(
#include <deep_ep/impls/afd/combine_send.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&combine_send_impl<{},{},{},{},{},{},{},{},{},{},{},{}>);
}}
)",c.num_src_ranks,c.num_ranks,c.num_experts/c.num_ranks,c.num_topk,hidden,c.num_vec_cores,
        c.num_source_lanes,c.num_layers,qa,tile,ub,timeout)).string();
}
static std::string compile_combine_recv_only(const AFDContextLayout& c,int hidden,int chunk,int ub,int64_t timeout,bool round_step) {
    return jit->compile_without_load("afd_combine_recv",std::format(R"(
#include <deep_ep/impls/afd/combine_recv.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&combine_recv_impl<{},{},{},{},{},{},{},{}>);
}}
)",c.num_experts,c.num_ranks,c.num_topk,hidden,chunk,c.num_vec_cores,timeout,round_step?1:0)).string();
}
static std::string compile_ffn_scheduler_only(const AFDContextLayout& c,int q,int qa,int alignment,bool precheck) {
    return jit->compile_without_load("afd_ffn_scheduler",std::format(R"(
#define AFD_KERNEL_FFN_SCHEDULER_TU
#include <deep_ep/impls/afd/ffn_scheduler.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {{
 auto ptr=reinterpret_cast<void*>(&ffn_scheduler_impl<{},{},{},{},{},{},{},{},{}>);
}}
)",c.num_ranks,c.num_experts,c.num_topk,c.num_source_lanes,c.num_layers,q,qa,alignment,precheck?1:0)).string();
}
static std::string compile_compute_ready_only() {
    return jit->compile_without_load("afd_compute_ready",R"(
#define AFD_KERNEL_COMPUTE_READY_TU
#include <deep_ep/impls/afd/ffn_scheduler.hpp>
using namespace deep_ep::afd;
static void __instantiate_kernel() {
 auto ptr=reinterpret_cast<void*>(&mark_compute_ready_impl);
}
)").string();
}
}
// ### PATCH END: remaining AFD launchers
