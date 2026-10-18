#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "buffers/base.hpp"
// ### PATCH START: AFD sender binding
#include "buffers/afd.hpp"
// ### PATCH END: AFD sender binding
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
#include "comm/api.hpp"
#include "runtime/jit.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    using namespace deep_ep;

    m.doc() = "DeepEP communication buffers (Ascend)";
    config::register_apis(m);
    m.attr("topk_idx_t") = py::cast(torch::kInt64);

    py::class_<EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    comm::register_apis(m);
    base::register_apis(m);

    // ### PATCH START: AFD owner and unchanged dispatchSend Tensor signature
    py::class_<AFDBuffer,std::shared_ptr<AFDBuffer>>(m, "AFDBuffer")
        .def(py::init<int,const std::string&,const std::vector<int>&,const std::vector<int>&,
             const std::vector<std::array<uint32_t,6>>&,const std::vector<uint32_t>&,
             int,int,int,int,int,bool,int,int,int,uint64_t,uint64_t>())
        .def_property_readonly("context",&AFDBuffer::context)
        .def_property_readonly("control_buffer",&AFDBuffer::control_buffer)
        .def_property_readonly("round_ledger",&AFDBuffer::round_ledger)
        .def_property_readonly("workspace",&AFDBuffer::workspace)
        .def_property_readonly("send_status",&AFDBuffer::send_status)
        .def_property_readonly("payload",&AFDBuffer::payload)
        .def("memory_usage",&AFDBuffer::memory_usage)
        .def("destroy",&AFDBuffer::destroy)
        .def("dispatchSend",&AFDBuffer::dispatchSend)
        .def_static("create_local_receiver",&AFDBuffer::create_local_receiver)
        .def("prepare_recv",&AFDBuffer::prepare_recv,py::arg("q"),py::arg("q_alloc"),
             py::arg("expert_alignment"),py::arg("sf_column_major")=false)
        .def_property_readonly("active",&AFDBuffer::active)
        .def_property_readonly("execution_state",&AFDBuffer::execution_state)
        .def_property_readonly("recv_workspace",&AFDBuffer::recv_workspace)
        .def_property_readonly("recv_x",&AFDBuffer::recv_x)
        .def_property_readonly("recv_sf",&AFDBuffer::recv_sf)
        .def_property_readonly("recv_src_metadata",&AFDBuffer::recv_src_metadata)
        .def("dispatchRecv",&AFDBuffer::dispatchRecv)
        .def("prepare_ffn",&AFDBuffer::prepare_ffn,py::arg("precheck")=false)
        .def("prepare_combine_recv",&AFDBuffer::prepare_combine_recv)
        .def_property_readonly("scheduler_workspace",&AFDBuffer::scheduler_workspace)
        .def_property_readonly("completion",&AFDBuffer::completion)
        .def_property_readonly("psum_num_recv_tokens_per_expert",&AFDBuffer::psum_num_recv_tokens_per_expert)
        .def_property_readonly("num_unaligned_recv_tokens_per_expert",&AFDBuffer::num_unaligned_recv_tokens_per_expert)
        .def_property_readonly("combine_x",&AFDBuffer::combine_x)
        .def_property_readonly("combine_send_workspace",&AFDBuffer::combine_send_workspace)
        .def_property_readonly("combine_recv_workspace",&AFDBuffer::combine_recv_workspace)
        .def_property_readonly("recv_status",&AFDBuffer::recv_status)
        .def("ffnScheduler",&AFDBuffer::ffnScheduler)
        .def("computeReady",&AFDBuffer::computeReady)
        .def("combineSend",&AFDBuffer::combineSend)
        .def("combineRecv",&AFDBuffer::combineRecv);
    // ### PATCH END: AFD owner and unchanged dispatchSend Tensor signature

    py::class_<EPBuffer, BufferBase>(m, "EPBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, int, bool>())
        .def_readonly("context", &EPBuffer::context)
        .def_readonly("lb_storage", &EPBuffer::lb_storage)
        .def("lb_prefetch_weights", &EPBuffer::lb_prefetch_weights)
        .def("lb_reduce_grads", &EPBuffer::lb_reduce_grads)
        .def("dispatch", &EPBuffer::dispatch)
        .def("combine", &EPBuffer::combine);

    py::class_<PPBuffer, BufferBase>(m, "PPBuffer")
        .def(py::init<int, int, std::string, int64_t, int, bool, int, bool>())
        .def_readonly("context", &PPBuffer::context)
        .def_readonly("storage", &PPBuffer::storage)
        .def("send", &PPBuffer::send)
        .def("recv", &PPBuffer::recv);

    py::class_<EngramBuffer, BufferBase>(m, "EngramBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, bool>())
        .def_readonly("context", &EngramBuffer::context)
        .def("set_config", &EngramBuffer::set_config)
        .def("write", &EngramBuffer::write)
        .def("fetch", &EngramBuffer::fetch);

    py::class_<BucketBuffer, BufferBase>(m, "BucketBuffer")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<std::string>, int64_t, bool, int, bool>())
        .def_readonly("contexts", &BucketBuffer::contexts)
        .def_readonly("storage", &BucketBuffer::storage)
        .def("reduce_scatter", &BucketBuffer::reduce_scatter)
        .def("all_reduce", &BucketBuffer::all_reduce)
        .def("all_gather", &BucketBuffer::all_gather);

    m.def("calculate_ep_buffer_size", &EPBuffer::calculate_buffer_size);
    m.def("get_num_allocation_alignment", []() {
        return kNumAllocationAlignmentBytes;
    });
    m.def("get_num_tma_alignment", []() {
        return kNumTMAAlignmentBytes;
    });
    m.def("get_num_rdma_alignment", []() {
        return kNumRDMAAlignmentBytes;
    });
}
