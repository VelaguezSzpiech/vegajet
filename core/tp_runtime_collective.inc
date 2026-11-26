// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Included by the live KFD translation unit, also exercised by fake-ioctl tests.
bool Runtime::configure_collective(uint64_t elements, const tp::CollectiveOptions& options, std::string* err) {
    auto fail = [&](const std::string& text) {
        if (err) *err = "N-rank collective: " + text;
        abort_dispatches();
        return false;
    };
    if (!open_ || failed_) return fail("runtime not open or already failed");
    // No storage may be re-used/freed until actual queue completion is observed.
    for (size_t r = 0; r < devices_.size(); ++r)
        if (!wait_idle(r)) return fail("queues did not quiesce before configuration");
    tp::CollectivePlan plan;
    try {
        std::vector<tp::RuntimeDevice> selected;
        for (const auto& d : devices_)
            selected.push_back({d.node_id, d.drm_render_minor, gpusel::bdf(d.pci_domain, d.location_id)});
        collective_topology_ = tp::read_runtime_topology(gpusel::sysfs_root(), selected);
        // Validate owner overrides using the same M2 planner as CPU --dry-run.
        (void)tp::plan_topology(collective_topology_.peers, options);
        plan = tp::plan_collective(elements, collective_topology_.peers, options);
        if (elements > UINT32_MAX / sizeof(float)) return fail("element count exceeds transfer ABI");
    } catch (const std::exception& e) { return fail(e.what()); }
    for (auto& inbox : collective_inboxes_) {
        if (!free_buffer(inbox.payload) || !free_buffer(inbox.flags) ||
            !free_buffer(inbox.host_payload) || !free_buffer(inbox.host_flags)) return fail("inbox cleanup failed");
    }
    collective_inboxes_.clear();
    if (!free_buffer(collective_routes_) || !free_buffer(collective_status_)) return fail("descriptor cleanup failed");
    collective_plan_ = plan;
    collective_options_ = options;
    collective_epoch_ = 0;
    if (plan.bypass) return true; // no RX, flags, routes, status or FP16 conversion
    const size_t n = devices_.size();
    for (size_t r = 0; r < n; ++r)
        if (!get_kernel(r, "vj_tp_allreduce_fp16")) return fail("vj_tp_collective.hsaco is required");
    collective_inboxes_.resize(n);
    const uint64_t payload_bytes = align_up(plan.rx_payload_bytes_per_rank, 4096ULL);
    const uint64_t flag_bytes = align_up(plan.rx_flag_bytes_per_rank, 4096ULL);
    for (size_t d = 0; d < n; ++d) {
        auto& inbox = collective_inboxes_[d];
        // Each owner has independent dual-BAR0 BOs: NC/coherent payload, UC epoch.
        if (!alloc_vram(d, payload_bytes, KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, inbox.payload) ||
            !alloc_vram(d, flag_bytes, KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED, inbox.flags))
            return fail("owner-local VRAM mailbox allocation failed");
        std::vector<uint32_t> writers{devices_[d].gpu_id};
        bool host = false;
        for (size_t s = 0; s < n; ++s) {
            if (plan.routes[s][d] == tp::Transport::PeerPush) writers.push_back(devices_[s].gpu_id);
            if (plan.routes[s][d] == tp::Transport::HostBounce) host = true;
        }
        if (!map_to_gpus(inbox.payload, writers) || !map_to_gpus(inbox.flags, writers))
            return fail("reported peer edge is not mappable; no silent fallback");
        std::vector<uint8_t> zero(static_cast<size_t>(flag_bytes), 0);
        if (!upload_to_vram(d, inbox.flags.va, zero.data(), zero.size())) return fail("epoch initialization failed");
        // Non-peer pairs use disjoint owner RX frames in coherent pinned host
        // memory. A producer pushes there; only the destination polls/reads it.
        // No remote reads or peer atomics, and no CPU relay in the token loop.
        if (host && (!alloc_userptr_pinned(payload_bytes, inbox.host_payload, true) ||
                     !alloc_userptr_pinned(flag_bytes, inbox.host_flags, true)))
            return fail("host-bounce allocation/map failed");
    }
    const uint64_t table_bytes = uint64_t(n) * n * sizeof(tp::CollectiveRoute);
    if (table_bytes / sizeof(tp::CollectiveRoute) / n != n ||
        !alloc_userptr_pinned(table_bytes, collective_routes_, true) ||
        !alloc_userptr_pinned(uint64_t(n) * 64, collective_status_, true))
        return fail("route/status allocation failed");
    std::vector<tp::MailboxAddresses> boxes;
    for (const auto& inbox : collective_inboxes_)
        boxes.push_back({inbox.payload.va, inbox.flags.va, inbox.host_payload.va, inbox.host_flags.va});
    try {
        const auto table = tp::collective_route_table(plan, boxes);
        std::memcpy(collective_routes_.host_ptr, table.data(), static_cast<size_t>(table_bytes));
    } catch (const std::exception& e) { return fail(e.what()); }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return true;
}

bool Runtime::collective_allreduce(const std::vector<uint64_t>& partials, const std::vector<uint64_t>& outputs,
                                  uint64_t elements, std::string* err) {
    auto fail = [&](const std::string& text) {
        if (err) *err = "N-rank collective: " + text;
        abort_dispatches(); // sticky fail-stop; caller must discard ALL tentative outputs
        return false;
    };
    if (!open_ || failed_ || partials.size() != devices_.size() || outputs.size() != devices_.size() ||
        std::any_of(partials.begin(), partials.end(), [](uint64_t p) { return !p || (p & 3); }) ||
        std::any_of(outputs.begin(), outputs.end(), [](uint64_t p) { return !p || (p & 3); }))
        return fail("invalid runtime/rank buffers");
    if (collective_plan_.elements != elements || collective_plan_.ranks != devices_.size())
        if (!configure_collective(elements, collective_options_, err)) return false;
    if (collective_plan_.bypass) {
        if (partials[0] == outputs[0]) return true;
        const auto* copy = get_kernel(0, "kfd_memcpy_u32");
        struct { uint64_t dst, src, words; } args{outputs[0], partials[0], elements};
        if (!copy || !dispatch_sync(0, *copy, static_cast<uint32_t>((elements + 255) / 256 * 256), 1, 1,
                                    256, 1, 1, &args, sizeof(args))) return fail("bypass copy failed");
        return true;
    }
    if (collective_epoch_ == UINT32_MAX) return fail("epoch exhausted; recreate runtime after quiescence");
    const uint32_t epoch = ++collective_epoch_;
    const uint32_t n = static_cast<uint32_t>(devices_.size());
    for (uint32_t r = 0; r < n; ++r) {
        tp::CollectiveArgs args{partials[r], outputs[r], collective_routes_.va + uint64_t(r) * n * sizeof(tp::CollectiveRoute),
                                collective_status_.va + uint64_t(r) * 64, elements, collective_plan_.payload_stride,
                                n, r, epoch, 1000000};
        const auto* kernel = get_kernel(r, "vj_tp_allreduce_fp16");
        if (!kernel || !enqueue_kernel(r, *kernel, 64, 1, 1, 64, 1, 1, &args, sizeof(args))) return fail("enqueue failed");
    }
    // Submit EVERY producer before waiting for any consumer. Two parity banks
    // and completion of every consumer before the next call enforce RT-B1 reuse.
    for (uint32_t r = 0; r < n; ++r) if (!submit_batch(r)) return fail("submit failed");
    for (uint32_t r = 0; r < n; ++r) if (!wait_idle(r)) return fail("timeout waiting for rank " + std::to_string(r));
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const auto* status = static_cast<const volatile uint32_t*>(collective_status_.host_ptr);
    for (uint32_t r = 0; r < n; ++r)
        if (status[r * 16]) return fail("rank " + std::to_string(r) + " protocol status=" + std::to_string(status[r * 16]));
    return true;
}
