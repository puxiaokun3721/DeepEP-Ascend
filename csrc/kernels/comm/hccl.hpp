#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <optional>

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_res.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_ep/common/math.hpp>
#include <deep_ep/common/exception.hpp>

#include "symmetric.hpp"

namespace deep_ep {

struct HCCLContext {
    std::shared_ptr<symmetric::SymmetricMemory> memory;
    HcclComm comm = nullptr;
    void* buffer = nullptr;
    int64_t num_bytes = 0;

    int num_ranks = 0;
    int rank_idx = 0;
    std::vector<int64_t> peer_ptrs;
    // ### PATCH START: AFD independent registration lengths
    std::vector<uint64_t> peer_buffer_bytes;
    bool allow_asymmetric_buffers = false;
    // AFD-private opt-in: world>2 AFD contexts use the endpoint-group channel table.
    bool afd_channel_groups = false;
    // ### PATCH END: AFD independent registration lengths

    // HCCL registration for UB_MEM address discovery
    HcclMemHandle mem_handle = nullptr;

    // Jetty endpoints, peer channels and their device lookup table
    // Endpoint-group channel stride: 0 = legacy single-endpoint table (multi-Jetty
    // owners share one effective endpoint); N = N channels per non-self peer.
    uint32_t channel_stride = 0;
    std::vector<EndpointHandle> jetty_endpoints;
    std::vector<ChannelHandle> channel_handles;
    void* jetty_device_ptrs = nullptr;  // Device copy of `channel_handles`

    // AFD extends construction with an opt-in final parameter. Default preserves EP.
    // The opt-in permits unequal peer allocations and requests the SQ depth used by the encoder.
    HCCLContext(const std::string& hccl_group_name, const int& rank_idx,
                const int& num_ranks, const int64_t& num_bytes,
                const int& num_jetties, const int& num_cpu_timeout_secs, const bool& use_huge1g_page,
                const std::shared_ptr<symmetric::SymmetricMemory>& shared_memory = nullptr,
                const bool allow_asymmetric_buffers = false,
                const bool afd_channel_groups = false):
        num_bytes(num_bytes),
        num_ranks(num_ranks), rank_idx(rank_idx) {
        EP_HOST_ASSERT(num_bytes > 0);
        EP_HOST_ASSERT(num_ranks > 0 and rank_idx >= 0 and rank_idx < num_ranks);
        // ### PATCH START: AFD resource options
        this->allow_asymmetric_buffers = allow_asymmetric_buffers;
        this->afd_channel_groups = afd_channel_groups;
        peer_buffer_bytes.resize(num_ranks, 0);
        peer_buffer_bytes[rank_idx] = num_bytes;
        // ### PATCH END: AFD resource options


        // TODO: relationships between layer and Jetties
        EP_HOST_ASSERT(num_jetties > 0);

        // All ranks must create buffers in the same order within each group.
        static std::unordered_map<std::string, int> context_idx_per_group;
        const auto buffer_idx = context_idx_per_group[hccl_group_name] ++;
        const auto buffer_uid = "deep_ep/" + hccl_group_name + "/" + std::to_string(buffer_idx);

        // Share the allocation when registering the same storage with another group.
        memory = shared_memory == nullptr ? std::make_shared<symmetric::SymmetricMemory>(num_bytes, use_huge1g_page) : shared_memory;
        EP_HOST_ASSERT(memory->num_bytes == num_bytes);
        buffer = memory->ptr;

        // Register buffer, make it visible to peers
        // TODO(HUAWEI): the class naming is so normal without namespaces
        CommMem mem;
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(this->num_bytes);
        HCCL_CHECK(HcclCommGetHandleWithName(hccl_group_name.c_str(), &comm));
        HCCL_CHECK(HcclCommMemReg(comm, buffer_uid.c_str(), &mem, &mem_handle));
        // TODO: do we need a barrier after member reg?

        // Discover peer buffer pointers
        peer_ptrs.resize(num_ranks, 0);
        peer_ptrs[rank_idx] = reinterpret_cast<int64_t>(buffer);

        if (num_ranks > 1) {
            EP_HOST_ASSERT(num_ranks % 2 == 0);
            // Constructor-only repair candidate: explicit layer 0 for UBMEM and Jetties
            uint32_t* layers = nullptr;
            uint32_t num_layers = 0;
            HCCL_CHECK(HcclRankGraphGetLayers(comm, &layers, &num_layers));
            EP_HOST_ASSERT(num_layers > 1);
            // HCCL owns `layers` and subsequent calls may invalidate it, so copy the layer value immediately.
            // Copy/check the explicit layer ID before any later SDK call.
            constexpr uint32_t kDiagnosticTransportLayer = 0;
            bool has_ubmem_layer = false;
            for (uint32_t i = 0; i < num_layers; ++i)
                has_ubmem_layer = has_ubmem_layer or layers[i] == kDiagnosticTransportLayer;
            EP_HOST_ASSERT(has_ubmem_layer);

            // Both transport paths use the checked layer ID; protocols remain unchanged.
            acquire_ubmem_channels(kDiagnosticTransportLayer);
            acquire_jetties(kDiagnosticTransportLayer, num_jetties, buffer_uid, num_cpu_timeout_secs);
        }
    }

    ~HCCLContext() noexcept(false) {
        // TODO: do we have to destroy the connections as well?
        // Teardown stage markers (diagnostic): rank/idx/stage/monotonic time/rc only.
        const auto hccl_destroy_t0 = std::chrono::steady_clock::now().time_since_epoch();
        auto hccl_stage = [&](const char* name, int idx, int rc) {
            const auto t_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch() - hccl_destroy_t0).count();
            std::fprintf(stderr, "AFD_HCCL_DESTROY rank=%d stage=%s idx=%d rc=%d t_us=%lld\n",
                         rank_idx, name, idx, rc, static_cast<long long>(t_us));
            std::fflush(stderr);
        };
        hccl_stage("table_free_begin", -1, 0);
        if (jetty_device_ptrs != nullptr) {
            const auto rc = aclrtFree(jetty_device_ptrs);
            hccl_stage("table_free_end", -1, static_cast<int>(rc));
            ACL_CHECK(rc);
            jetty_device_ptrs = nullptr;
        } else hccl_stage("table_free_end", -1, 0);
        hccl_stage("channel_destroy_begin", -1, 0);
        if (not channel_handles.empty()) {
            const auto rc = HcommChannelDestroy(channel_handles.data(), static_cast<uint32_t>(channel_handles.size()));
            hccl_stage("channel_destroy_end", -1, static_cast<int>(rc));
            HCOMM_CHECK(rc);
        } else hccl_stage("channel_destroy_end", -1, 0);
        for (size_t i = 0; i < jetty_endpoints.size(); ++ i) {
            hccl_stage("mem_unreg_begin", static_cast<int>(i), 0);
            const auto unreg_rc = HcommMemUnreg(jetty_endpoints[i], jetty_mem_handles[i]);
            hccl_stage("mem_unreg_end", static_cast<int>(i), static_cast<int>(unreg_rc));
            HCOMM_CHECK(unreg_rc);
            hccl_stage("endpoint_destroy_begin", static_cast<int>(i), 0);
            const auto ep_rc = HcommEndpointDestroy(jetty_endpoints[i]);
            hccl_stage("endpoint_destroy_end", static_cast<int>(i), static_cast<int>(ep_rc));
            HCOMM_CHECK(ep_rc);
        }
        hccl_stage("memory_reset_begin", -1, 0);
        memory.reset();
        hccl_stage("memory_reset_end", -1, 0);
        buffer = nullptr;
    }

private:
    std::vector<HcommMemHandle> jetty_mem_handles;
    // HCOMM shallow-copies channelName pointers; keep channel_names until channel destruction.
    std::vector<std::string> channel_names;

    // Build a channel desc for the first link to `peer_rank_idx` on `layer` whose
    // protocol satisfies `accept`. Returns nullptr if no such link exists.
    template <typename Accept>
    std::optional<HcclChannelDesc> make_peer_desc(uint32_t layer, int peer_rank_idx, Accept accept) {
        CommLink* links = nullptr;
        uint32_t num_links = 0;
        HCCL_CHECK(HcclRankGraphGetLinks(comm, layer, rank_idx, peer_rank_idx, &links, &num_links));
        for (uint32_t i = 0; i < num_links; ++ i) {
            const auto& link = links[i];
            const auto proto = link.linkAttr.linkProtocol;
            if (not accept(proto))
                continue;

            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank              = peer_rank_idx;
            desc.channelProtocol         = proto;
            desc.localEndpoint.protocol  = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr  = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc       = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc      = link.dstEndpointDesc.loc;
            // TODO: what do the magic numbers (3 and 1) mean?
            desc.notifyNum               = 3;
            // Attach our registered workspace so peers' writes land in our region.
            desc.memHandles              = &mem_handle;
            desc.memHandleNum            = 1;
            return desc;
        }
        return std::nullopt;
    }

    // Build descs for every non-self peer on `layer`, one per peer via the first
    // link matching `accept`. Asserts `err` if some peer has no matching link.
    template <typename Accept>
    std::vector<HcclChannelDesc> build_peer_descs(uint32_t layer, Accept accept, const char* err) {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(num_ranks - 1);
        for (int peer_rank_idx = 0; peer_rank_idx < num_ranks; ++ peer_rank_idx) {
            if (peer_rank_idx == rank_idx)
                continue;

            auto desc = make_peer_desc(layer, peer_rank_idx, accept);
            EP_HOST_ASSERT(desc.has_value() and err);
            descs.push_back(*desc);
        }
        return descs;
    }

    // One UBMEM channel per non-self peer on `layer`; resolves peer_ptrs to each
    // remote's registered workspace address.
    void acquire_ubmem_channels(uint32_t layer) {
        auto descs = build_peer_descs(layer,
            [](auto proto) { return proto == COMM_PROTOCOL_UB_MEM; },
            "no UB_MEM link to peer on the chosen layer");

        std::vector<ChannelHandle> channels(descs.size(), 0);
        HCCL_CHECK(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                                      descs.data(), static_cast<uint32_t>(descs.size()),
                                      channels.data()));

        // The last remote mem is the registered userbuffer.
        for (size_t i = 0; i < descs.size(); ++ i) {
            const auto peer_rank_idx = static_cast<int>(descs[i].remoteRank);
            uint32_t num_mems = 0;
            CommMem* remote_mems = nullptr;
            char** mem_tags = nullptr;
            HCCL_CHECK(HcclChannelGetRemoteMems(comm, channels[i], &num_mems, &remote_mems, &mem_tags));
            EP_HOST_ASSERT(num_mems >= 1);
            // ### PATCH START: AFD non-symmetric payload registration
            EP_HOST_ASSERT(allow_asymmetric_buffers || remote_mems[num_mems - 1].size == this->num_bytes);
            EP_HOST_ASSERT(remote_mems[num_mems - 1].size > 0);
            peer_buffer_bytes[peer_rank_idx] = remote_mems[num_mems - 1].size;
            // ### PATCH END: AFD non-symmetric payload registration
            peer_ptrs[peer_rank_idx] = reinterpret_cast<int64_t>(remote_mems[num_mems - 1].addr);
        }
    }

    // Acquire exactly one shared Jetty per AIV, with separate channels for peer metadata.
    void acquire_jetties(uint32_t layer, const int& num_jetties,
                         const std::string& buffer_uid, const int& num_cpu_timeout_secs) {
        const auto peer_descs = build_peer_descs(layer,
            [](auto proto) { return proto == COMM_PROTOCOL_UBC_CTP; },
            "no UBC_CTP link to peer on the chosen layer");

        // ### PATCH START: branch B endpoint-group channels (AFD private)
        // AFD-private opt-in. All AFD ranks in world>2 use the grouped table (uniform
        // per-rank channel counts); a single local group still uses the new layout.
        // world2 keeps the legacy table and non-AFD contexts never enter this branch.
        if (afd_channel_groups and num_ranks > 2) {
            acquire_grouped_jetties(num_jetties, buffer_uid, num_cpu_timeout_secs, peer_descs);
            return;
        }
        // ### PATCH END: branch B endpoint-group channels (AFD private)

        const auto num_peers = num_ranks - 1;
        const auto num_channels = math::max(num_jetties, num_peers);

        // Register the buffer on the HCOMM endpoint for UB_CTP; HCCL registration for UB_MEM.
        const auto& jetty_endpoint_desc = peer_descs.front().localEndpoint;
        // ### PATCH START: endpoint comparison diag (review-directed scalars only)
        {
            for (size_t di = 0; di < peer_descs.size(); ++ di) {
                const auto& pd = peer_descs[di];
                CommLink* dbg_links = nullptr;
                uint32_t dbg_num = 0;
                (void)HcclRankGraphGetLinks(comm, layer, rank_idx, static_cast<int>(pd.remoteRank), &dbg_links, &dbg_num);
                uint32_t n_ubc = 0;
                for (uint32_t li = 0; li < dbg_num; ++ li)
                    if (dbg_links != nullptr and dbg_links[li].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) ++ n_ubc;
                const auto& e = pd.localEndpoint;
                uint32_t adiff = 0, ldiff = 0;
                for (size_t bi = 0; bi < sizeof(e.commAddr.raws); ++ bi)
                    adiff += (e.commAddr.raws[bi] != jetty_endpoint_desc.commAddr.raws[bi]) ? 1u : 0u;
                for (size_t bi = 0; bi < sizeof(e.loc.raws); ++ bi)
                    ldiff += (e.loc.raws[bi] != jetty_endpoint_desc.loc.raws[bi]) ? 1u : 0u;
                const bool eq_proto = e.protocol == jetty_endpoint_desc.protocol;
                const bool eq_atype = e.commAddr.type == jetty_endpoint_desc.commAddr.type;
                const bool eq_araws = adiff == 0;
                const bool eq_ltype = e.loc.locType == jetty_endpoint_desc.loc.locType;
                const bool eq_lraws = ldiff == 0;
                std::fprintf(stderr, "ENDPOINT_DIAG rank=%d peer=%u layer=%u links=%u ubc_links=%u proto=%d rproto=%d atype=%d r_atype=%d ltype=%d r_ltype=%d eq_proto=%d eq_atype=%d eq_araws=%d eq_ltype=%d eq_lraws=%d adiff=%u ldiff=%u",
                    rank_idx, static_cast<unsigned>(pd.remoteRank), layer, dbg_num, n_ubc,
                    static_cast<int>(e.protocol), static_cast<int>(pd.remoteEndpoint.protocol),
                    static_cast<int>(e.commAddr.type), static_cast<int>(jetty_endpoint_desc.commAddr.type),
                    static_cast<int>(e.loc.locType), static_cast<int>(jetty_endpoint_desc.loc.locType),
                    eq_proto ? 1 : 0, eq_atype ? 1 : 0, eq_araws ? 1 : 0, eq_ltype ? 1 : 0, eq_lraws ? 1 : 0,
                    adiff, ldiff);
                std::fprintf(stderr, "%s", "");
                std::fflush(stderr);
            }
            std::fprintf(stderr, "ENDPOINT_DIAG_SUMMARY rank=%d num_ranks=%d num_peers=%zu%sz", rank_idx, num_ranks, peer_descs.size(), "");
            std::fflush(stderr);
        }
        // ### PATCH END: endpoint comparison diag

        // Each AIV uses one Jetty for all peers, so require same local endpoint.
        for (const auto& peer_desc: peer_descs) {
            const auto& endpoint = peer_desc.localEndpoint;
            EP_HOST_ASSERT(endpoint.protocol == COMM_PROTOCOL_UBC_CTP and peer_desc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP);
            EP_HOST_ASSERT(endpoint.protocol == jetty_endpoint_desc.protocol and
                           endpoint.commAddr.type == jetty_endpoint_desc.commAddr.type and
                           std::memcmp(endpoint.commAddr.raws, jetty_endpoint_desc.commAddr.raws, sizeof(endpoint.commAddr.raws)) == 0 and
                           endpoint.loc.locType == jetty_endpoint_desc.loc.locType and
                           std::memcmp(endpoint.loc.raws, jetty_endpoint_desc.loc.raws, sizeof(endpoint.loc.raws)) == 0 and
                           "multiple local UBC_CTP endpoints are not supported");
        }
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(num_bytes);
        jetty_endpoints.resize(num_jetties);
        jetty_mem_handles.resize(num_jetties);
        for (int i = 0; i < num_jetties; ++ i) {
            HCOMM_CHECK(HcommEndpointCreate(&jetty_endpoint_desc, &jetty_endpoints[i]));
            HCOMM_CHECK(HcommMemReg(jetty_endpoints[i], buffer_uid.c_str(), &mem, &jetty_mem_handles[i]));
        }

        // Allocate a private listen port; port 0 in a channel desc otherwise selects the fixed port 60001.
        uint32_t listen_port = 0;
        HCOMM_CHECK(HcommEndpointGetListenPort(jetty_endpoints.front(), &listen_port));
        EP_HOST_ASSERT(listen_port > 0 and listen_port <= std::numeric_limits<uint16_t>::max());
        const auto listen_ports = gather_listen_ports(static_cast<uint16_t>(listen_port), num_cpu_timeout_secs);

        // The first entries describe every peer; pair the remaining Jetty owners directly.
        HcommChannelConfig config = nullptr;
        HCOMM_CHECK(HcommChannelConfigCreate(&config));
        HCOMM_CHECK(HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1));
        channel_handles.resize(num_channels);
        channel_names.resize(num_channels);
        for (int channel_idx = 0; channel_idx < num_channels; ++ channel_idx) {
            int peer_idx = channel_idx, pair_idx = 0;
            if (channel_idx >= num_peers) {
                const auto peer_rank_idx = rank_idx ^ 1;
                peer_idx = peer_rank_idx < rank_idx ? peer_rank_idx : peer_rank_idx - 1;
                pair_idx = channel_idx - num_peers + 1;
            }
            const auto& peer_desc = peer_descs[peer_idx];
            const auto peer_rank_idx = static_cast<int>(peer_desc.remoteRank);
            channel_names[channel_idx] = buffer_uid + "/" +
                std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                std::to_string(std::max(rank_idx, peer_rank_idx)) + "/" + std::to_string(pair_idx);
            EP_HOST_ASSERT(channel_names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
            HcommChannelDesc desc;
            HCOMM_CHECK(HcommChannelDescInit(&desc, 1));
            desc.remoteEndpoint = peer_desc.remoteEndpoint;
            desc.notifyNum = 3;  // same as HCCL
            desc.memHandles = &jetty_mem_handles[channel_idx % num_jetties];
            desc.memHandleNum = 1;
            // The lower rank listens; the higher rank connects socket.
            desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = listen_ports[std::min(rank_idx, peer_rank_idx)];
            desc.channelName = channel_names[channel_idx].c_str();
            // Leave desc.qos at 0xFFFFFFFF to use HCOMM's default UB QoS (4).
            // Keep the HCOMM descriptor default. API depth is not the device WQEBB count.
            // Constructor-only: verify actual queue metadata before device queue writers run.
            HCOMM_CHECK(HcommChannelCreateWithConfig(jetty_endpoints[channel_idx % num_jetties], CommEngine::COMM_ENGINE_AIV,
                                                     &desc, 1, config, channel_handles.data() + channel_idx));
        }
        HCOMM_CHECK(HcommChannelConfigDestroy(config));
        wait_for_channels(num_cpu_timeout_secs);

        // Device code indexes this table by both AIV and non-self peer.
        // Keep all peer entries visible even when peers outnumber AIV-owned queues.
        const auto num_table_bytes = sizeof(ChannelHandle) * channel_handles.size();
        ACL_CHECK(aclrtMalloc(&jetty_device_ptrs, num_table_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(jetty_device_ptrs, num_table_bytes,
                              channel_handles.data(), num_table_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ### PATCH START: branch B endpoint-group channels (AFD private)
    static bool same_effective_addr(const CommAddr& x, const CommAddr& y) {
        if (x.type != y.type)
            return false;
        switch (x.type) {
            case COMM_ADDR_TYPE_IP_V4: return std::memcmp(&x.addr, &y.addr, sizeof(x.addr)) == 0;
            case COMM_ADDR_TYPE_IP_V6: return std::memcmp(&x.addr6, &y.addr6, sizeof(x.addr6)) == 0;
            case COMM_ADDR_TYPE_ID: return x.id == y.id;
            case COMM_ADDR_TYPE_EID: return std::memcmp(x.eid, y.eid, sizeof(x.eid)) == 0;
            default: return std::memcmp(x.raws, y.raws, sizeof(x.raws)) == 0;
        }
    }
    static bool same_effective_loc(const EndpointLoc& x, const EndpointLoc& y) {
        if (x.locType != y.locType)
            return false;
        switch (x.locType) {
            case ENDPOINT_LOC_TYPE_DEVICE: return x.device.devPhyId == y.device.devPhyId and
                                                  x.device.superDevId == y.device.superDevId and
                                                  x.device.serverIdx == y.device.serverIdx and
                                                  x.device.superPodIdx == y.device.superPodIdx;
            case ENDPOINT_LOC_TYPE_HOST: return x.host.id == y.host.id;
            default: return std::memcmp(x.raws, y.raws, sizeof(x.raws)) == 0;
        }
    }
    // Per-(peer, jetty) port gather; frees its temporary device buffers on any throw.
    std::vector<uint16_t> gather_per_peer_ports(const std::vector<uint16_t>& own_ports,
                                                const int& num_peers, const int& num_jetties,
                                                const int& num_cpu_timeout_secs) const {
        const auto count = static_cast<uint32_t>(num_peers * num_jetties);
        const auto num_bytes = sizeof(uint16_t) * size_t(count);
        std::vector<uint16_t> ports(size_t(num_ranks) * count, 0);
        void* device_all = nullptr;
        void* device_own = nullptr;
        bool gather_pending = false;
        try {
            ACL_CHECK(aclrtMalloc(&device_all, num_bytes * size_t(num_ranks), ACL_MEM_MALLOC_NORMAL_ONLY));
            ACL_CHECK(aclrtMalloc(&device_own, num_bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
            ACL_CHECK(aclrtMemcpy(device_own, num_bytes, own_ports.data(), num_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            const auto stream = c10_npu::getCurrentNPUStream();
            gather_pending = true;
            HCCL_CHECK(HcclAllGather(device_own, device_all, count, HCCL_DATA_TYPE_UINT16, comm, stream));
            ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, num_cpu_timeout_secs * 1000));
            gather_pending = false;
            ACL_CHECK(aclrtMemcpy(ports.data(), num_bytes * size_t(num_ranks), device_all,
                                  num_bytes * size_t(num_ranks), ACL_MEMCPY_DEVICE_TO_HOST));
        } catch (...) {
            if (gather_pending) {
                // Completion is unknown: keep the buffers alive until task teardown.
                std::fprintf(stderr, "AFD_GROUP_CLEANUP gather_pending=1 buffers_retained=1\n");
            } else {
                if (device_own != nullptr) {
                    const auto rc = aclrtFree(device_own);
                    std::fprintf(stderr, "AFD_GROUP_CLEANUP own_free_rc=%d\n", static_cast<int>(rc));
                }
                if (device_all != nullptr) {
                    const auto rc = aclrtFree(device_all);
                    std::fprintf(stderr, "AFD_GROUP_CLEANUP all_free_rc=%d\n", static_cast<int>(rc));
                }
            }
            throw;
        }
        const auto own_free_rc = aclrtFree(device_own);
        const auto all_free_rc = aclrtFree(device_all);
        std::fprintf(stderr, "AFD_GROUP_CLEANUP own_free_rc=%d all_free_rc=%d\n",
                     static_cast<int>(own_free_rc), static_cast<int>(all_free_rc));
        ACL_CHECK(own_free_rc);
        ACL_CHECK(all_free_rc);
        for (const auto port: ports)
            EP_HOST_ASSERT(port > 0);
        return ports;
    }
    // Reverse-order release of only successfully created resources. Returns whether
    // every release call reported success; never clears state or retries.
    bool cleanup_grouped_resources(const size_t created_channels, const size_t registered_mems,
                                   const size_t created_endpoints, HcommChannelConfig config) {
        // Stop on failed release: dependent registrations/endpoints may still be live.
        // The caller propagates failure and the bounded task runner ends this process.
        if (created_channels > 0) {
            const auto rc = HcommChannelDestroy(channel_handles.data(), static_cast<uint32_t>(created_channels));
            std::fprintf(stderr, "AFD_GROUP_CLEANUP channel_rc=%d\n", static_cast<int>(rc));
            if (rc != HCCL_SUCCESS) return false;
        }
        for (size_t k = registered_mems; k > 0; -- k) {
            const size_t slot = k - 1;
            const auto rc = HcommMemUnreg(jetty_endpoints[slot], jetty_mem_handles[slot]);
            std::fprintf(stderr, "AFD_GROUP_CLEANUP mem_slot=%zu rc=%d\n", slot, static_cast<int>(rc));
            if (rc != HCCL_SUCCESS) return false;
        }
        for (size_t k = created_endpoints; k > 0; -- k) {
            const auto rc = HcommEndpointDestroy(jetty_endpoints[k - 1]);
            std::fprintf(stderr, "AFD_GROUP_CLEANUP endpoint_slot=%zu rc=%d\n", k - 1, static_cast<int>(rc));
            if (rc != HCCL_SUCCESS) return false;
        }
        if (config != nullptr) {
            const auto rc = HcommChannelConfigDestroy(config);
            std::fprintf(stderr, "AFD_GROUP_CLEANUP config_rc=%d\n", static_cast<int>(rc));
            if (rc != HCCL_SUCCESS) return false;
        }
        if (jetty_device_ptrs != nullptr) {
            const auto rc = aclrtFree(jetty_device_ptrs);
            std::fprintf(stderr, "AFD_GROUP_CLEANUP table_free_rc=%d\n", static_cast<int>(rc));
            if (rc != ACL_SUCCESS) return false;
            jetty_device_ptrs = nullptr;
        }
        return true;
    }
    // AFD world>2 always uses this layout (uniform per-rank counts). One local group
    // still gets one channel per (peer, AIV); real descriptors are never merged.
    void acquire_grouped_jetties(const int& num_jetties, const std::string& buffer_uid,
                                 const int& num_cpu_timeout_secs,
                                 const std::vector<HcclChannelDesc>& peer_descs) {
        const auto num_peers = num_ranks - 1;
        std::vector<int> group_of(num_peers, -1);
        std::vector<EndpointDesc> group_reps;
        for (int p = 0; p < num_peers; ++ p) {
            const auto& e = peer_descs[p].localEndpoint;
            int g = -1;
            for (size_t k = 0; k < group_reps.size(); ++ k)
                if (e.protocol == group_reps[k].protocol and
                    same_effective_addr(e.commAddr, group_reps[k].commAddr) and
                    same_effective_loc(e.loc, group_reps[k].loc)) {
                    g = static_cast<int>(k);
                    break;
                }
            if (g < 0) {
                g = static_cast<int>(group_reps.size());
                group_reps.push_back(e);
            }
            group_of[p] = g;
        }
        const int num_groups = static_cast<int>(group_reps.size());
        channel_stride = static_cast<uint32_t>(num_jetties);
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(num_bytes);
        jetty_endpoints.assign(size_t(num_groups) * num_jetties, 0);
        jetty_mem_handles.assign(size_t(num_groups) * num_jetties, 0);
        HcommChannelConfig config = nullptr;
        size_t created_endpoints = 0, registered_mems = 0, created_channels = 0;
        try {
            for (int g = 0; g < num_groups; ++ g) {
                for (int i = 0; i < num_jetties; ++ i) {
                    const size_t slot = size_t(g) * num_jetties + i;
                    HCOMM_CHECK(HcommEndpointCreate(&group_reps[g], &jetty_endpoints[slot]));
                    ++ created_endpoints;
                    HCOMM_CHECK(HcommMemReg(jetty_endpoints[slot], buffer_uid.c_str(), &mem, &jetty_mem_handles[slot]));
                    ++ registered_mems;
                }
            }
            std::vector<uint16_t> own_ports(size_t(num_peers) * num_jetties, 0);
            for (int p = 0; p < num_peers; ++ p) {
                for (int i = 0; i < num_jetties; ++ i) {
                    uint32_t port = 0;
                    HCOMM_CHECK(HcommEndpointGetListenPort(jetty_endpoints[size_t(group_of[p]) * num_jetties + i], &port));
                    EP_HOST_ASSERT(port > 0 and port <= std::numeric_limits<uint16_t>::max());
                    own_ports[size_t(p) * num_jetties + i] = static_cast<uint16_t>(port);
                }
            }
            const auto all_ports = gather_per_peer_ports(own_ports, num_peers, num_jetties, num_cpu_timeout_secs);
            HCOMM_CHECK(HcommChannelConfigCreate(&config));
            HCOMM_CHECK(HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1));
            const size_t num_channels = size_t(num_peers) * num_jetties;
            channel_handles.assign(num_channels, 0);
            channel_names.assign(num_channels, std::string());
            for (int p = 0; p < num_peers; ++ p) {
                const auto& peer_desc = peer_descs[p];
                const auto peer_rank_idx = static_cast<int>(peer_desc.remoteRank);
                for (int i = 0; i < num_jetties; ++ i) {
                    const size_t channel_idx = size_t(p) * num_jetties + i;
                    channel_names[channel_idx] = buffer_uid + "/" +
                        std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                        std::to_string(std::max(rank_idx, peer_rank_idx)) + "/" + std::to_string(i);
                    EP_HOST_ASSERT(channel_names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
                    HcommChannelDesc desc;
                    HCOMM_CHECK(HcommChannelDescInit(&desc, 1));
                    desc.remoteEndpoint = peer_desc.remoteEndpoint;
                    desc.notifyNum = 3;
                    desc.memHandles = &jetty_mem_handles[size_t(group_of[p]) * num_jetties + i];
                    desc.memHandleNum = 1;
                    desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
                    if (rank_idx < peer_rank_idx) {
                        desc.port = own_ports[channel_idx];
                    } else {
                        const uint16_t server_rank = static_cast<uint16_t>(peer_rank_idx);
                        const uint16_t pos_of_me = static_cast<uint16_t>(rank_idx - 1);
                        desc.port = all_ports[(size_t(server_rank) * num_peers + pos_of_me) * num_jetties + i];
                    }
                    desc.channelName = channel_names[channel_idx].c_str();
                    HCOMM_CHECK(HcommChannelCreateWithConfig(
                        jetty_endpoints[size_t(group_of[p]) * num_jetties + i], CommEngine::COMM_ENGINE_AIV,
                        &desc, 1, config, channel_handles.data() + channel_idx));
                    ++ created_channels;
                }
            }
            const auto config_destroy_rc = HcommChannelConfigDestroy(config);
            config = nullptr;  // Do not retry a release with an unknown outcome.
            HCOMM_CHECK(config_destroy_rc);
            wait_for_channels(num_cpu_timeout_secs);
            const auto num_table_bytes = sizeof(ChannelHandle) * channel_handles.size();
            ACL_CHECK(aclrtMalloc(&jetty_device_ptrs, num_table_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            ACL_CHECK(aclrtMemcpy(jetty_device_ptrs, num_table_bytes,
                                  channel_handles.data(), num_table_bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
        } catch (const std::exception& error) {
            const bool cleaned = cleanup_grouped_resources(created_channels, registered_mems, created_endpoints, config);
            if (!cleaned)
                throw std::runtime_error(std::string("endpoint-group cleanup failed; original error: ") + error.what());
            throw;
        } catch (...) {
            const bool cleaned = cleanup_grouped_resources(created_channels, registered_mems, created_endpoints, config);
            if (!cleaned)
                throw std::runtime_error("endpoint-group cleanup failed with unknown original error");
            throw;
        }
    }
    // ### PATCH END: branch B endpoint-group channels (AFD private)

    std::vector<uint16_t> gather_listen_ports(const uint16_t& listen_port, const int& num_cpu_timeout_secs) const {
        // Exchange once per buffer, regardless of the number of peers or Jetties.
        std::vector<uint16_t> listen_ports(num_ranks);
        const auto num_port_bytes = sizeof(uint16_t) * num_ranks;
        void* device_ports = nullptr;
        ACL_CHECK(aclrtMalloc(&device_ports, num_port_bytes + sizeof(uint16_t), ACL_MEM_MALLOC_NORMAL_ONLY));
        const auto stream = c10_npu::getCurrentNPUStream();
        const auto local_port = math::advance_ptr<void>(device_ports, num_port_bytes);
        ACL_CHECK(aclrtMemcpy(local_port, sizeof(listen_port), &listen_port, sizeof(listen_port), ACL_MEMCPY_HOST_TO_DEVICE));
        HCCL_CHECK(HcclAllGather(local_port, device_ports, 1, HCCL_DATA_TYPE_UINT16, comm, stream));
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, num_cpu_timeout_secs * 1000));
        ACL_CHECK(aclrtMemcpy(listen_ports.data(), num_port_bytes, device_ports, num_port_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtFree(device_ports));
        for (const auto& port: listen_ports)
            EP_HOST_ASSERT(port > 0);
        return listen_ports;
    }

    void wait_for_channels(const int& num_cpu_timeout_secs) const {
        // State values from the `HcommChannelGetStatus` contract.
        // TODO(HUAWEI): do not hardcode
        static constexpr int kChannelReady = 0;
        static constexpr int kChannelConnecting = 1;
        static constexpr int kChannelFailed = 2;
        static constexpr int kChannelTimeout = 3;

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(num_cpu_timeout_secs);
        std::vector<int32_t> statuses(channel_handles.size(), kChannelConnecting);
        while (true) {
            const auto result = HcommChannelGetStatus(channel_handles.data(), static_cast<uint32_t>(channel_handles.size()), statuses.data());
            if (result != HCCL_E_AGAIN)
                HCOMM_CHECK(result);

            if (result == HCCL_SUCCESS) {
                bool all_ready = true;
                for (const int& status: statuses) {
                    if (status == kChannelReady)
                        continue;

                    all_ready = false;
                    EP_HOST_ASSERT(status != kChannelFailed and status != kChannelTimeout);
                }
                if (all_ready)
                    return;
            }
            EP_HOST_ASSERT(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
};

// ### PATCH START: rank graph read-only probe (review-directed scalars only)
inline std::string rank_graph_probe_scalars(const std::string& hccl_group_name, uint32_t layer) {
    HcclComm comm = nullptr;
    const auto rc_comm = HcclCommGetHandleWithName(hccl_group_name.c_str(), &comm);
    uint32_t rank = 0, num_ranks = 0;
    const auto rc_rank = HcclGetRankId(comm, &rank);
    const auto rc_size = HcclGetRankSize(comm, &num_ranks);
    uint32_t* layers = nullptr;
    uint32_t num_layers = 0;
    const auto rc_layers = HcclRankGraphGetLayers(comm, &layers, &num_layers);
    bool layer_present = false;
    for (uint32_t i = 0; i < num_layers; ++ i)
        layer_present = layer_present or layers[i] == layer;
    struct PeerSnap {
        int valid = 0;
        HcclResult rc = HCCL_E_PTR;
        uint32_t n_links = 0, n_ubc = 0;
        CommProtocol proto = COMM_PROTOCOL_RESERVED;
        CommProtocol rproto = COMM_PROTOCOL_RESERVED;
        CommAddrType atype = COMM_ADDR_TYPE_RESERVED;
        EndpointLocType ltype = ENDPOINT_LOC_TYPE_RESERVED;
        uint8_t addr16[16] = {0};
        uint8_t addr20[20] = {0};
        uint8_t araw36[36] = {0};
        uint8_t lraw60[60] = {0};
        uint32_t dev[4] = {0, 0, 0, 0};
    };
    std::vector<uint32_t> peers;
    std::vector<PeerSnap> snaps;
    for (uint32_t peer = 0; peer < num_ranks; ++ peer) {
        if (peer == rank) continue;
        PeerSnap sn;
        CommLink* links = nullptr;
        uint32_t n = 0;
        sn.rc = HcclRankGraphGetLinks(comm, layer, rank, peer, &links, &n);
        sn.n_links = n;
        if (sn.rc == HCCL_SUCCESS) {
            const CommLink* chosen = nullptr;
            for (uint32_t i = 0; i < n; ++ i) {
                if (links[i].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                    ++ sn.n_ubc;
                    if (chosen == nullptr) chosen = &links[i];
                }
            }
            if (chosen != nullptr) {
                sn.valid = 1;
                sn.proto = chosen->srcEndpointDesc.protocol;
                sn.rproto = chosen->dstEndpointDesc.protocol;
                sn.atype = chosen->srcEndpointDesc.commAddr.type;
                sn.ltype = chosen->srcEndpointDesc.loc.locType;
                std::memcpy(sn.araw36, chosen->srcEndpointDesc.commAddr.raws, 36);
                std::memcpy(sn.addr16, sn.araw36, 16);
                std::memcpy(sn.addr20, sn.araw36 + 16, 20);
                std::memcpy(sn.lraw60, chosen->srcEndpointDesc.loc.raws, 60);
                if (sn.ltype == ENDPOINT_LOC_TYPE_DEVICE) {
                    sn.dev[0] = chosen->srcEndpointDesc.loc.device.devPhyId;
                    sn.dev[1] = chosen->srcEndpointDesc.loc.device.superDevId;
                    sn.dev[2] = chosen->srcEndpointDesc.loc.device.serverIdx;
                    sn.dev[3] = chosen->srcEndpointDesc.loc.device.superPodIdx;
                }
            }
        }
        peers.push_back(peer);
        snaps.push_back(sn);
    }
    auto diff_cnt = [](const uint8_t* x, const uint8_t* y, size_t n) {
        uint32_t c = 0;
        for (size_t i = 0; i < n; ++ i) c += (x[i] != y[i]) ? 1u : 0u;
        return c;
    };
    auto eff_eq = [](const PeerSnap& x, const PeerSnap& y) -> bool {
        if (x.valid == 0 or y.valid == 0) return false;
        return x.proto == y.proto and x.atype == y.atype and x.ltype == y.ltype and
               std::memcmp(x.addr16, y.addr16, 16) == 0 and
               x.dev[0] == y.dev[0] and x.dev[1] == y.dev[1] and
               x.dev[2] == y.dev[2] and x.dev[3] == y.dev[3];
    };
    auto raw_eq = [](const PeerSnap& x, const PeerSnap& y) -> bool {
        if (x.valid == 0 or y.valid == 0) return false;
        return x.proto == y.proto and x.atype == y.atype and x.ltype == y.ltype and
               std::memcmp(x.araw36, y.araw36, 36) == 0 and
               std::memcmp(x.lraw60, y.lraw60, 60) == 0;
    };
    std::string out = "{\"entry\":\"rank_graph_probe\",";
    char buf[512];
    std::snprintf(buf, sizeof(buf), "\"rank\":%u,\"num_ranks\":%u,\"layer\":%u,\"rc_comm\":%d,\"rc_rank\":%d,\"rc_size\":%d,\"rc_layers\":%d,\"num_layers\":%u,\"layer_present\":%d,",
        rank, num_ranks, layer, static_cast<int>(rc_comm), static_cast<int>(rc_rank), static_cast<int>(rc_size), static_cast<int>(rc_layers), num_layers, layer_present ? 1 : 0);
    out += buf;
    std::snprintf(buf, sizeof(buf), "\"sizeof_addr6\":%d,\"sizeof_raws\":%d,\"sizeof_loc_raws\":%d,",
        static_cast<int>(sizeof(((CommAddr*)nullptr)->addr6)),
        static_cast<int>(sizeof(((CommAddr*)nullptr)->raws)),
        static_cast<int>(sizeof(((EndpointLoc*)nullptr)->raws)));
    out += buf;
    out += "\"peers\":[";
    for (size_t i = 0; i < peers.size(); ++ i) {
        const auto& sn = snaps[i];
        if (i) out += ",";
        std::snprintf(buf, sizeof(buf), "{\"peer\":%u,\"rc_links\":%d,\"links\":%u,\"ubc_links\":%u,\"valid\":%d",
            peers[i], static_cast<int>(sn.rc), sn.n_links, sn.n_ubc, sn.valid);
        out += buf;
        if (sn.valid) {
            std::snprintf(buf, sizeof(buf), ",\"proto\":%d,\"rproto\":%d,\"atype\":%d,\"ltype\":%d",
                static_cast<int>(sn.proto), static_cast<int>(sn.rproto),
                static_cast<int>(sn.atype), static_cast<int>(sn.ltype));
            out += buf;
            if (i > 0 and snaps[0].valid) {
                const auto& f = snaps[0];
                std::snprintf(buf, sizeof(buf), ",\"eq_addr16\":%d,\"diff16\":%u,\"eq_addr20\":%d,\"diff20\":%u,\"eq_raw36\":%d,\"eq_lraw60\":%d,\"eq_ltype\":%d,\"eq_dev\":[%d,%d,%d,%d]",
                    std::memcmp(sn.addr16, f.addr16, 16) == 0 ? 1 : 0, diff_cnt(sn.addr16, f.addr16, 16),
                    std::memcmp(sn.addr20, f.addr20, 20) == 0 ? 1 : 0, diff_cnt(sn.addr20, f.addr20, 20),
                    std::memcmp(sn.araw36, f.araw36, 36) == 0 ? 1 : 0,
                    std::memcmp(sn.lraw60, f.lraw60, 60) == 0 ? 1 : 0,
                    sn.ltype == f.ltype ? 1 : 0,
                    sn.dev[0] == f.dev[0] ? 1 : 0, sn.dev[1] == f.dev[1] ? 1 : 0,
                    sn.dev[2] == f.dev[2] ? 1 : 0, sn.dev[3] == f.dev[3] ? 1 : 0);
                out += buf;
            }
        }
        out += "}";
    }
    out += "],\"pairwise\":[";
    bool first_pair = true;
    for (size_t i = 0; i < snaps.size(); ++ i) {
        for (size_t j = i + 1; j < snaps.size(); ++ j) {
            if (!first_pair) out += ",";
            first_pair = false;
            std::snprintf(buf, sizeof(buf), "{\"a\":%u,\"b\":%u,\"eff_eq\":%d,\"raw_eq\":%d}",
                peers[i], peers[j], eff_eq(snaps[i], snaps[j]) ? 1 : 0, raw_eq(snaps[i], snaps[j]) ? 1 : 0);
            out += buf;
        }
    }
    out += "],\"groups\":[";
    std::vector<int> gid(snaps.size(), -1);
    int next_gid = 0;
    for (size_t i = 0; i < snaps.size(); ++ i) {
        if (snaps[i].valid == 0) continue;
        int g = -1;
        for (size_t k = 0; k < i; ++ k)
            if (gid[k] >= 0 and eff_eq(snaps[i], snaps[k])) { g = gid[k]; break; }
        if (g < 0) g = next_gid ++;
        gid[i] = g;
    }
    for (size_t i = 0; i < snaps.size(); ++ i) {
        if (i) out += ",";
        std::snprintf(buf, sizeof(buf), "%d", gid[i]);
        out += buf;
    }
    out += "]}" + std::string(1, static_cast<char>(10));
    return out;
}
// ### PATCH END: rank graph read-only probe
}  // namespace deep_ep
