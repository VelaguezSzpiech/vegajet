// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Included after the existing upload helpers. The TP=2 implementation below this
// include is deliberately separate; none of its kernel ABIs or launch order change.
namespace {
struct RawLayerState {
    LayerWeightsDev w;
    uint64_t state = 0, conv = 0, kc = 0, vc = 0;
};
struct RawScratch {
    uint64_t h, xn, xq, xd, partial, reduced, proj, conv_out, decay, beta;
    uint64_t aq, ad, q, gate, logits, shgu, shq, shd, router, ids, probs, sg;
    uint64_t midq, midd, down, shared, map, argmax;
    Q8WeightDev vocab;
    uint64_t final_norm;
    std::vector<RawLayerState> layers;
};

OrnithBenchMetrics run_ornith_nrank(GgufFile& f, BpeTokenizer& tokenizer, const std::vector<int32_t>& prompt_ids, const OrnithGpuOptions& opt) {
    OrnithBenchMetrics result;
    std::vector<int32_t> generated;
    const auto now_sec = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    try {
        const std::string denied = ornith_raw_admit(opt);
        if (!denied.empty()) throw std::runtime_error(denied);
        if (opt.decode_tokens < 0) throw std::runtime_error("generic Ornith: negative decode token count");
        if (prompt_ids.empty()) throw std::runtime_error("generic Ornith: nonempty prompt IDs required");
        if (prompt_ids.size() + uint64_t(opt.decode_tokens) > std::min(opt.max_ctx, 8192u))
            throw std::runtime_error("generic Ornith: context exceeds 8192");
        const uint32_t context = opt.serve_hook ? opt.max_ctx : static_cast<uint32_t>(prompt_ids.size() + opt.decode_tokens);
        LoadOptions lo;
        lo.context_length = context;
        // Model metadata validation only; the live allocator uses RawLayout below.
        lo.gpus.assign(opt.tp, GpuTarget{60, 1, 1, 1ULL << 50});
        OrnithModel model;
        const auto report = model.load(f, lo);
        if (!report.ok) throw std::runtime_error(report.error);
        if (dot8_promote_enabled()) {
            // v1.3 dot8-wire: this path uploads Gate+Up as merged IQ2_S and Down as exact DOT8 only.
            Dot8KernelRoutes routes;
            routes.down_dot8 = true;  // vj_moe_dot8 is a required module below; a missing kernel throws at launch
            Dot8UploadPlan dot8_plan;
            std::string dot8_error;
            if (!resolve_dot8_upload_plan(model.tensor_plan(), model.hparams().n_trunk, false, routes, dot8_plan, &dot8_error))
                throw std::runtime_error("generic Ornith: " + dot8_error);
        }
        const auto& hp = model.hparams();
        if (!opt.placement_json.empty()) placement::bind_executor(placement::from_json(opt.placement_json), opt.tp);
        kfd::Runtime rt;
        std::string error;
        if (!rt.init(30.5, 64, &error, opt.tp)) throw std::runtime_error(error);
        if (rt.devices().size() != opt.tp) throw std::runtime_error("generic Ornith: selected device count must equal --tp");
        std::vector<tp::DeviceProfile> devices;
        for (const auto& d : rt.devices()) {
            tp::DeviceProfile p;
            char bdf[32];
            std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", d.pci_domain, d.location_id >> 8, (d.location_id >> 3) & 31, d.location_id & 7);
            p.bdf = bdf; p.n_cu = d.n_cu; p.mclk = 1;
            const auto key = tp::device_key(p.bdf, gpusel::sysfs_root());
            if (key.rfind("uid:", 0) == 0) p.unique_id = key.substr(4);
            p.bw_hbm_gbs = opt.tp_bandwidths.empty() ? 1 : opt.tp_bandwidths.at(devices.size());
            p.usable_vram_bytes = rt.vram_arena(devices.size()).capacity() - rt.vram_arena(devices.size()).used();
            devices.push_back(p);
        }
        const auto shard_options = opt.tp_shard_json.empty() ? tp::ShardOptions{} : tp::shard_options_from_json(opt.tp_shard_json);
        if (!opt.placement_json.empty()) {
            const auto config = placement::from_json(opt.placement_json);
            const auto ordered = placement::executor_devices(config, devices);
            for (size_t r = 0; r < devices.size(); ++r)
                if (devices[r].bdf != ordered[r].bdf) throw std::runtime_error("placement executor rank order mismatch");
            placement::validate(config, devices, hp.n_trunk, [&](const auto& ranks, uint32_t, uint32_t) {
                (void)raw_layout(hp, context, ranks, shard_options);
            });
        }
        const auto layout = raw_layout(hp, context, devices, shard_options);
        // Explicit module allow-list: unrelated experimental artifacts never become dependencies.
        for (const char* module : {"vj_kfd_base", "vj_tp_collective", "vj_ornith_ar", "vj_ornith_gemm",
                                   "vj_ornith_attn", "vj_ornith_moe", "vj_ornith_moe_m1", "vj_moe_dot8", "vj_moe_asym", "vj_ornith_vocab", "vj_ornith_nrank"})
            if (!rt.load_code_object((opt.hsaco_dir.empty() ? std::string(VEGAJET_DEFAULT_HSACO_DIR) : opt.hsaco_dir) + "/" + module + ".hsaco", &error)) throw std::runtime_error(error);
        tp::CollectiveOptions collective;
        if (!opt.tp_collective_json.empty()) collective = tp::collective_options_from_json(opt.tp_collective_json);
        if (!rt.configure_collective(3072, collective, &error)) throw std::runtime_error(error);
        auto check = [&](bool ok) { if (!ok) throw std::runtime_error(error.empty() ? "generic Ornith: runtime operation failed" : error); };
        auto launch = [&](size_t rank, const char* name, uint32_t groups, uint32_t threads, const auto& args) {
            if (!groups) return;
            const auto* kernel = rt.get_kernel(rank, name);
            if (!kernel) throw std::runtime_error(std::string("generic Ornith: missing kernel ") + name);
            check(rt.enqueue_kernel(rank, *kernel, groups * threads, 1, 1, threads, 1, 1, &args, sizeof(args), true));
        };
        auto flush = [&]() {
            for (size_t r : layout.rank_order) check(rt.submit_batch(r));
            for (size_t r : layout.rank_order) check(rt.wait_idle(r));
        };
        auto alloc = [&](size_t r, uint64_t bytes) {
            const auto va = rt.vram_arena(r).alloc(std::max<uint64_t>(bytes, 256), 256);
            if (!va) throw std::runtime_error("generic Ornith: rank " + std::to_string(r) + " VRAM allocation failed");
            return va;
        };
        auto zero = [&](size_t r, uint64_t va, uint64_t bytes) {
            struct { uint64_t dst; uint32_t value, pad; uint64_t count; } args{va, 0, 0, bytes / 4};
            launch(r, "kfd_memset_u32", static_cast<uint32_t>((args.count + 255) / 256), 256, args);
        };
        auto q4 = [&](size_t r, const std::vector<RowSource>& rows, uint32_t first, uint32_t cols,
                      const std::vector<uint32_t>* indices, Q4A8WeightDev& dst) {
            const uint64_t count = indices ? indices->size() : cols;
            if (!count || count % 32) throw std::runtime_error("generic Ornith: unaligned dense shard");
            for (const auto& source : rows) {
                const auto* ti = f.tensor(source.tensor_name);
                if (!ti || ti->n_dims != 2 || !ti->ne[0] || !ti->ne[1])
                    throw std::runtime_error("generic Ornith: invalid matrix " + source.tensor_name);
                const auto* format = quant::DecoderRegistry::instance().find(static_cast<quant::DiskQuantType>(ti->type));
                if (!format || !format->decode || !format->block_elems || ti->ne[0] % format->block_elems)
                    throw std::runtime_error("generic Ornith: unsupported quant " + source.tensor_name);
                for (auto row : source.row_indices) if (row >= ti->ne[1])
                    throw std::runtime_error("generic Ornith: row map outside " + source.tensor_name);
                if (indices) {
                    for (auto col : *indices) if (col >= ti->ne[0])
                        throw std::runtime_error("generic Ornith: column map outside " + source.tensor_name);
                } else if (uint64_t(first) + cols > ti->ne[0])
                    throw std::runtime_error("generic Ornith: column range outside " + source.tensor_name);
            }
            check(upload_fused_q4a8_tcc(rt, r, f, rows, first, cols, indices, dst, &error));
        };
        auto floats = [&](size_t r, const std::string& name, const std::vector<uint32_t>* rows, uint64_t& dst) {
            const auto* ti = f.tensor(name);
            if (!ti || ti->type != static_cast<uint32_t>(GgmlType::F32) || !ti->n_elems())
                throw std::runtime_error("generic Ornith: requires F32 tensor " + name);
            const uint64_t width = ti->n_dims >= 2 ? ti->ne[0] : 1;
            if (!width) throw std::runtime_error("generic Ornith: empty float row " + name);
            uint64_t expected = 0;
            if (name.find("ssm_conv1d.weight") != std::string::npos) expected = 12288 * 4;
            else if (name.find("ssm_a") != std::string::npos || name.find("ssm_dt.bias") != std::string::npos) expected = 64;
            else if (name.find("ssm_norm.weight") != std::string::npos) expected = 128;
            else if (name.find("attn_q_norm.weight") != std::string::npos || name.find("attn_k_norm.weight") != std::string::npos) expected = 256;
            else expected = 3072;
            if (ti->n_elems() != expected || (name.find("ssm_conv1d.weight") != std::string::npos && width != 4))
                throw std::runtime_error("generic Ornith: float tensor geometry " + name);
            if (rows) for (auto row : *rows) if (row >= ti->n_elems() / width)
                throw std::runtime_error("generic Ornith: float row map outside " + name);
            check(upload_f32_rows(rt, r, f, name, rows, dst, &error));
        };
        std::vector<RawScratch> ranks(opt.tp);
        const auto all = iota_range(0, 3072);
        const auto* output = f.tensor("output.weight") ? f.tensor("output.weight") : f.tensor("token_embd.weight");
        if (!output || output->n_dims != 2 || output->ne[0] != 3072 || output->ne[1] != hp.vocab)
            throw std::runtime_error("generic Ornith: vocabulary matrix geometry");
        for (size_t r : layout.rank_order) {
            auto& s = ranks[r]; const auto& map = layout.ranks[r];
            const auto& b = map.scratch;
            const uint32_t ff = map.ff.size(), sf = map.shared_ff.size();
            // Upper geometry bounds keep all temporary aliases independent of rank partition.
            s.h = alloc(r, b.hidden); s.xn = alloc(r, b.hidden); s.xq = alloc(r, b.xq); s.xd = alloc(r, b.xd);
            s.partial = alloc(r, b.hidden); s.reduced = alloc(r, b.hidden);
            s.proj = alloc(r, b.proj); s.conv_out = alloc(r, b.conv_out);
            s.decay = alloc(r, b.gates); s.beta = alloc(r, b.gates);
            s.aq = alloc(r, b.aq); s.ad = alloc(r, b.ad);
            s.q = alloc(r, b.attn); s.gate = alloc(r, b.attn);
            s.logits = alloc(r, b.logits); s.argmax = alloc(r, 8);
            s.shgu = alloc(r, b.shared_gu); s.shq = alloc(r, b.shared_q); s.shd = alloc(r, b.shared_ds);
            s.router = alloc(r, 257 * 4); s.ids = alloc(r, 8 * 4); s.probs = alloc(r, 8 * 4); s.sg = alloc(r, 4);
            s.midq = alloc(r, b.moe_q); s.midd = alloc(r, b.moe_ds);
            s.down = alloc(r, 4 * 3072 * 4); s.shared = alloc(r, b.hidden);
            s.map = alloc(r, b.head_map);
            if (!map.attn.q_to_local_kv.empty()) check(rt.upload_to_vram(r, s.map, map.attn.q_to_local_kv.data(), map.attn.q_to_local_kv.size() * 4));
            if (map.vocab.size()) check(upload_q8_tcc(rt, r, f, f.tensor("output.weight") ? "output.weight" : "token_embd.weight",
                map.vocab.begin, map.vocab.size(), s.vocab, &error));
            floats(r, "output_norm.weight", nullptr, s.final_norm);
            s.layers.resize(hp.n_trunk);
            for (uint32_t il = 0; il < hp.n_trunk; ++il) {
                auto& state = s.layers[il]; auto& w = state.w;
                const std::string p = "blk." + std::to_string(il) + ".";
                w.is_gdn = hp.is_recurrent(il);
                floats(r, p + "attn_norm.weight", nullptr, w.attn_norm_va);
                floats(r, p + "post_attention_norm.weight", nullptr, w.post_attn_norm_va);
                if (w.is_gdn && !map.groups.empty()) {
                    const auto values = expand_heads(map.v_heads, 128);
                    q4(r, {{p + "attn_qkv.weight", map.qkv_rows}, {p + "attn_gate.weight", values},
                           {p + "ssm_alpha.weight", map.v_heads}, {p + "ssm_beta.weight", map.v_heads}}, 0, 3072, nullptr, w.w_gdn_qkv_gab);
                    q4(r, {{p + "ssm_out.weight", all}}, 0, 0, &values, w.w_ssm_out);
                    floats(r, p + "ssm_conv1d.weight", &map.conv_rows, w.ssm_conv_w_va);
                    floats(r, p + "ssm_a", &map.v_heads, w.ssm_a_va);
                    floats(r, p + "ssm_dt.bias", &map.v_heads, w.ssm_dt_va);
                    floats(r, p + "ssm_norm.weight", nullptr, w.ssm_norm_va);
                    state.state = alloc(r, map.state_bytes); state.conv = alloc(r, map.conv_bytes);
                    zero(r, state.state, map.state_bytes); zero(r, state.conv, map.conv_bytes);
                    flush();
                } else if (!w.is_gdn && !map.attn.q_heads.empty()) {
                    const auto columns = expand_heads(map.attn.q_heads, 256);
                    q4(r, {{p + "attn_q.weight", map.q_rows}, {p + "attn_k.weight", map.kv_rows},
                           {p + "attn_v.weight", map.kv_rows}}, 0, 3072, nullptr, w.w_attn_qkv);
                    q4(r, {{p + "attn_output.weight", all}}, 0, 0, &columns, w.w_o);
                    floats(r, p + "attn_q_norm.weight", nullptr, w.q_norm_va);
                    floats(r, p + "attn_k_norm.weight", nullptr, w.k_norm_va);
                    state.kc = alloc(r, map.kv_bytes / 2); state.vc = alloc(r, map.kv_bytes / 2);
                }
                const auto* router = f.tensor(p + "ffn_gate_inp.weight");
                const auto* gate = f.tensor(p + "ffn_gate_inp_shexp.weight");
                if (!router || !gate || router->type != static_cast<uint32_t>(GgmlType::F32) || router->ne[0] != 3072 || router->ne[1] != 256 || router->n_elems() != 256 * 3072 || gate->n_elems() != 3072)
                    throw std::runtime_error("generic Ornith: router geometry");
                check(upload_f32_as_f16(rt, r, f, p + "ffn_gate_inp.weight", w.w_router_f16_va, &error));
                floats(r, p + "ffn_gate_inp_shexp.weight", nullptr, w.w_sh_gate_va);
                if (sf) {
                    const auto rows = iota_range(map.shared_ff.begin, sf);
                    q4(r, {{p + "ffn_gate_shexp.weight", rows}, {p + "ffn_up_shexp.weight", rows}}, 0, 3072, nullptr, w.w_sh_gu);
                    q4(r, {{p + "ffn_down_shexp.weight", all}}, map.shared_ff.begin, sf, nullptr, w.w_sh_down);
                }
                if (ff) {
                    const auto* tg = f.tensor(p + "ffn_gate_exps.weight");
                    const auto* tu = f.tensor(p + "ffn_up_exps.weight");
                    const auto* td = f.tensor(p + "ffn_down_exps.weight");
                    check(validate_expert_tensor(tg, p + "ffn_gate_exps.weight", 3072, 1024, 256, false, true, &error));
                    check(validate_expert_tensor(tu, p + "ffn_up_exps.weight", 3072, 1024, 256, false, true, &error));
                    check(validate_expert_tensor(td, p + "ffn_down_exps.weight", 3072, 1024, 256, true, true, &error));
                    const auto* rg = f.tensor_data(*tg); const auto* ru = f.tensor_data(*tu); const auto* rd = f.tensor_data(*td);
                    if (!rg || !ru || !rd) throw std::runtime_error("generic Ornith: missing expert payload");
            auto pack_row = [&](const uint8_t* g_row, const uint8_t* u_row, uint32_t* dst) {
                uint32_t* qp  = dst;
                uint32_t* hs2 = dst + 96u * 4u;
                uint32_t* d2  = dst + 120u * 4u;
                for (uint32_t b = 0; b < 12u; ++b) {
                    const uint8_t* gb = g_row + b * 82u;
                    const uint8_t* ub = u_row + b * 82u;
                    uint16_t gd = 0, ud = 0;
                    std::memcpy(&gd, gb, 2);
                    std::memcpy(&ud, ub, 2);
                    d2[b] = static_cast<uint32_t>(gd) | (static_cast<uint32_t>(ud) << 16);
                    for (uint32_t sb = 0; sb < 8u; ++sb) {
                        const uint32_t ib = b * 8u + sb;
                        const uint8_t* gq = gb + 2u + sb * 4u;
                        const uint8_t* gs = gb + 34u + sb * 4u;
                        const uint8_t* uq = ub + 2u + sb * 4u;
                        const uint8_t* us = ub + 34u + sb * 4u;
                        qp[ib * 4u + 0u] = uint32_t(gq[0]) | (uint32_t(gs[0]) << 8) | (uint32_t(gq[1]) << 16) | (uint32_t(gs[1]) << 24);
                        qp[ib * 4u + 1u] = uint32_t(gq[2]) | (uint32_t(gs[2]) << 8) | (uint32_t(gq[3]) << 16) | (uint32_t(gs[3]) << 24);
                        qp[ib * 4u + 2u] = uint32_t(uq[0]) | (uint32_t(us[0]) << 8) | (uint32_t(uq[1]) << 16) | (uint32_t(us[1]) << 24);
                        qp[ib * 4u + 3u] = uint32_t(uq[2]) | (uint32_t(us[2]) << 8) | (uint32_t(uq[3]) << 16) | (uint32_t(us[3]) << 24);
                        hs2[ib] = uint32_t(gb[66u + sb]) | (uint32_t(gb[74u + sb]) << 8) |
                                  (uint32_t(ub[66u + sb]) << 16) | (uint32_t(ub[74u + sb]) << 24);
                    }
                }
            };
                    std::vector<uint32_t> gu(uint64_t(256) * ff * 492);
                    for (uint32_t e = 0; e < 256; ++e) for (uint32_t i = 0; i < ff; ++i) {
                        const uint64_t src = uint64_t(e) * 1024 + map.ff.begin + i;
                        pack_row(rg + src * 984, ru + src * 984, gu.data() + (uint64_t(e) * ff + i) * 492);
                    }
                    w.w_gu_iq2s_va = alloc(r, gu.size() * 4);
                    check(rt.upload_to_vram(r, w.w_gu_iq2s_va, gu.data(), gu.size() * 4));
                    const auto down = dot8_iq3_asym_layout(256ULL * 3072, ff);
                    std::vector<uint8_t> codes(down.w_plane_bytes()), scales(down.s_plane_bytes());
                    check(dot8_repack_iq3xxs_asym_rows(rd, 392, map.ff.begin, down, 0, 256ULL * 3072, codes.data(), scales.data(), &error));
                    w.w_dn_d8_w_va = alloc(r, codes.size());
                    w.w_dn_d8_s_va = down.scale_embedded ? w.w_dn_d8_w_va + down.s_offset_bytes : alloc(r, scales.size());
                    w.dn_d8_w_stride = down.w_row_stride_bytes; w.dn_d8_s_stride = down.s_row_stride_bytes;
                    check(rt.upload_to_vram(r, w.w_dn_d8_w_va, codes.data(), codes.size()));
                    if (!scales.empty()) check(rt.upload_to_vram(r, w.w_dn_d8_s_va, scales.data(), scales.size()));
                }
            }
        }
        const auto* embedding = f.tensor("token_embd.weight");
        if (!embedding || embedding->ne[0] != 3072 || embedding->ne[1] != hp.vocab || embedding->type != static_cast<uint32_t>(GgmlType::Q8_0))
            throw std::runtime_error("generic Ornith: Q8_0 embedding geometry unsupported");
        const auto* embedding_data = f.tensor_data(*embedding);
        if (!embedding_data) throw std::runtime_error("generic Ornith: embedding payload missing");
        const auto& decoder = quant::DecoderRegistry::instance();
        const auto embedding_type = static_cast<quant::DiskQuantType>(embedding->type);
        const auto embedding_stride = decoder.row_bytes(embedding_type, 3072);
        std::vector<uint64_t> partials, outputs;
        for (const auto& s : ranks) { partials.push_back(s.partial); outputs.push_back(s.reduced); }
        const float eps = static_cast<float>(hp.rms_eps);
        auto norm = [&](size_t r, uint64_t weight, uint64_t add) {
            auto& s = ranks[r];
            struct { uint64_t h, add, w, xn, xq, xd; uint32_t m, d; float eps; }
                a{s.h, add, weight, s.xn, s.xq, s.xd, 1, 3072, eps};
            launch(r, "ornith_rms_quant_q8_kernel", 1, 256, a);
        };
        auto gemv = [&](size_t r, const Q4A8WeightDev& w, uint64_t xq, uint64_t xd, uint64_t out) {
            struct { uint64_t wq, wd, xq, xd, out; uint32_t rows, cols, stride, pad; uint64_t peer; }
                a{w.qs_va, w.dm_va, xq, xd, out, w.rows, w.cols, w.row_stride_bytes, 0, 0};
            launch(r, "ornith_gemv_q4a8_m1_w32_kernel", (w.rows + 7) / 8, 256, a);
        };
        // v1.3 M7 --profile-stages: per decode token, kfd_stamp_clock_u64 (vj_kfd_base) brackets every stage on every
        // rank, and the collective kernel stamps its own entry / flags posted / peers seen / exit
        // (Runtime::collective_stamps). Only stamps of one GPU are subtracted; core/rank_timing does the split.
        const uint32_t n_stamps = 4u * hp.n_trunk + 2u;
        std::vector<uint64_t> stamp_va(opt.tp, 0);
        std::vector<std::vector<uint64_t>> ar_stamp(opt.tp);  // 4 stamps per AllReduce, 2 AllReduces per layer
        std::vector<std::vector<uint8_t>> ar_stamped(opt.tp);
        ranktiming::Aggregator rank_timer(opt.tp);
        uint32_t timed_tokens = 0;
        bool timing_on = false;
        if (opt.profile_stages)
            for (size_t r = 0; r < opt.tp; ++r) {
                stamp_va[r] = alloc(r, uint64_t(n_stamps) * 8);
                ar_stamp[r].assign(uint64_t(hp.n_trunk) * 8, 0);
                ar_stamped[r].assign(uint64_t(hp.n_trunk) * 2, 0);
            }
        auto stamp = [&](size_t r, uint32_t idx) {
            if (!timing_on) return;
            struct { uint64_t dst; uint32_t idx, pad; } a{stamp_va[r], idx, 0};
            launch(r, "kfd_stamp_clock_u64", 1, 64, a);
        };
        auto ar_mark = [&](uint32_t slot) {  // right after collective_allreduce: the status slot holds this call's stamps
            if (!timing_on) return;
            for (size_t r = 0; r < opt.tp; ++r) ar_stamped[r][slot] = rt.collective_stamps(r, &ar_stamp[r][uint64_t(slot) * 4]);
        };
        auto book_token = [&]() {  // after the token's last flush, before any host read-back of this token
            if (!timing_on) return;
            std::string terr;
            for (size_t r = 0; r < opt.tp; ++r) {
                std::vector<uint64_t> s(n_stamps);
                check(rt.download_from_vram(r, s.data(), stamp_va[r], s.size() * 8));
                const auto rank = static_cast<uint32_t>(r);
                auto add = [&](const char* stage, uint64_t b, uint64_t e, uint64_t w) {
                    if (!rank_timer.add(rank, stage, b, e, w, true, &terr)) throw std::runtime_error("generic Ornith: rank timing: " + terr);
                };
                auto ar = [&](uint32_t slot) {
                    const uint64_t* t = &ar_stamp[r][uint64_t(slot) * 4];
                    if (ar_stamped[r][slot]) add("allreduce", t[0], t[3], t[2] - t[1]);
                    else rank_timer.break_chain(rank);  // bypass copy (N=1) or no stamps: the interval is unstamped work
                };
                rank_timer.break_chain(rank);  // token boundary: embedding upload / argmax / stamp read-back kernels
                for (uint32_t il = 0; il < hp.n_trunk; ++il) {
                    add("mixer", s[4 * il], s[4 * il + 1], 0);
                    ar(2 * il);
                    add("moe", s[4 * il + 2], s[4 * il + 3], 0);
                    ar(2 * il + 1);
                }
                add("final_norm+vocab", s[4 * hp.n_trunk], s[4 * hp.n_trunk + 1], 0);
            }
            ++timed_tokens;
        };
        auto step = [&](int32_t token, uint32_t position) {
            if (token < 0 || uint64_t(token) >= hp.vocab || position >= context)
                throw std::runtime_error("generic Ornith: token/context out of range");
            std::vector<float> h(3072);
            decoder.decode_row_f32(embedding_type, embedding_data + uint64_t(token) * embedding_stride, h.data(), 3072);
            for (size_t r : layout.rank_order) check(rt.upload_to_vram(r, ranks[r].h, h.data(), h.size() * 4));
            for (uint32_t il = 0; il < hp.n_trunk; ++il) {
                for (size_t r : layout.rank_order) {
                    auto& s = ranks[r]; auto& state = s.layers[il]; const auto& w = state.w; const auto& m = layout.ranks[r];
                    stamp(r, 4 * il);
                    norm(r, w.attn_norm_va, il ? s.reduced : 0);
                    if (w.is_gdn && !m.groups.empty()) {
                        const uint32_t nk = m.groups.size(), nv = m.v_heads.size(), cd = m.conv_rows.size();
                        gemv(r, w.w_gdn_qkv_gab, s.xq, s.xd, s.proj);
                        struct { uint64_t in, cw, cr, co, dt, sa, decay, beta; uint32_t m, nk, nv, cd, stride, ao, bo, head, ring; float eps; }
                            conv{s.proj, w.ssm_conv_w_va, state.conv, s.conv_out, w.ssm_dt_va, w.ssm_a_va, s.decay, s.beta,
                                 1, nk, nv, cd, w.w_gdn_qkv_gab.rows, cd + nv * 128, cd + nv * 129, 0, 0, eps};
                        launch(r, "ornith_gdn_conv_l2_gate_kernel", 2 * nk + nv + 1, 128, conv);
                        struct { uint64_t state, conv, in, decay, beta, norm, oq, od; uint32_t m, nk, nv, cd, stride, zo, head, ring; float eps; }
                            recur{state.state, s.conv_out, s.proj, s.decay, s.beta, w.ssm_norm_va, s.aq, s.ad,
                                  1, nk, nv, cd, w.w_gdn_qkv_gab.rows, cd, 0, 0, eps};
                        launch(r, "ornith_gdn_recur_quant_kernel", nv, 1024, recur);
                        gemv(r, w.w_ssm_out, s.aq, s.ad, s.partial);
                    } else if (!w.is_gdn && !m.attn.q_heads.empty()) {
                        const uint32_t nq = m.attn.q_heads.size(), nk = m.attn.kv_heads.size();
                        gemv(r, w.w_attn_qkv, s.xq, s.xd, s.proj);
                        struct { uint64_t in, qnorm, knorm, q, gate, kc, vc; uint32_t m, pos, nq, nk, stride, rope; float base, eps; }
                            prep{s.proj, w.q_norm_va, w.k_norm_va, s.q, s.gate, state.kc, state.vc, 1, position, nq, nk,
                                 w.w_attn_qkv.rows, hp.rope_dims, static_cast<float>(hp.rope_base), eps};
                        launch(r, "ornith_attn_qkv_prep_kernel", nq + 2 * nk, 256, prep);
                        struct { uint64_t q, gate, kc, vc, oq, od; uint32_t m, pos, nq, nk; uint64_t map; }
                            flash{s.q, s.gate, state.kc, state.vc, s.aq, s.ad, 1, position, nq, nk, s.map};
                        launch(r, "ornith_attn_flash_nrank_kernel", nq, 256, flash);
                        gemv(r, w.w_o, s.aq, s.ad, s.partial);
                    } else zero(r, s.partial, 3072 * 4);
                    stamp(r, 4 * il + 1);
                }
                flush();
                check(rt.collective_allreduce(partials, outputs, 3072, &error));
                ar_mark(2 * il);
                for (size_t r : layout.rank_order) {
                    auto& s = ranks[r]; const auto& w = s.layers[il].w; const auto& map = layout.ranks[r];
                    stamp(r, 4 * il + 2);
                    const uint32_t ff = map.ff.size(), sf = map.shared_ff.size();
                    norm(r, w.post_attn_norm_va, s.reduced);
                    struct { uint64_t wr, wg, xn, logits; uint32_t m, d; }
                        router{w.w_router_f16_va, w.w_sh_gate_va, s.xn, s.router, 1, 3072};
                    launch(r, "ornith_moe_router_fp16_logits_kernel", 33, 256, router);
                    struct { uint64_t logits, ids, probs, sg; uint32_t m; } top{s.router, s.ids, s.probs, s.sg, 1};
                    launch(r, "ornith_moe_router_top8_kernel", 1, 256, top);
                    if (sf) {
                        gemv(r, w.w_sh_gu, s.xq, s.xd, s.shgu);
                        struct { uint64_t in, sg, oq, od; uint32_t m, ff; } sw{s.shgu, s.sg, s.shq, s.shd, 1, sf};
                        launch(r, "ornith_swiglu_quant_q8_kernel", 1, 256, sw);
                        gemv(r, w.w_sh_down, s.shq, s.shd, s.shared);
                    } else zero(r, s.shared, 3072 * 4);
                    if (ff) {
                        struct { uint64_t w, xq, xd, ids, oq, od; uint32_t m, ff; }
                            gu{w.w_gu_iq2s_va, s.xq, s.xd, s.ids, s.midq, s.midd, 1, ff};
                        launch(r, "ornith_moe_gu_iq2s_merged_kernel", 8 * (ff / 32), 256, gu);
                        struct { uint64_t wq, ws, mq, md, ids, probs, shared, out; uint32_t m, d, wst, sst, ff, pad; uint64_t peer; }
                            down{w.w_dn_d8_w_va, w.w_dn_d8_s_va, s.midq, s.midd, s.ids, s.probs, s.shared, s.partial,
                                 1, 3072, w.dn_d8_w_stride, w.dn_d8_s_stride, ff, 0, 0};
                        launch(r, "vj_moe_down_asym_kernel", 3072 / 16, 256, down);
                    } else {
                        struct { uint64_t dst, src, words; } copy{s.partial, s.shared, 3072};
                        launch(r, "kfd_memcpy_u32", 12, 256, copy);
                    }
                    stamp(r, 4 * il + 3);
                }
                flush();
                check(rt.collective_allreduce(partials, outputs, 3072, &error));
                ar_mark(2 * il + 1);
            }
            for (size_t r : layout.rank_order) {
                auto& s = ranks[r]; stamp(r, 4 * hp.n_trunk); norm(r, s.final_norm, s.reduced);
                if (layout.ranks[r].vocab.size()) {
                    struct { uint64_t wq, wd, xq, xd, out; uint32_t rows, cols, stride; }
                        v{s.vocab.qs_va, s.vocab.ds_va, s.xq, s.xd, s.logits, s.vocab.rows, 3072, s.vocab.row_stride_bytes};
                    launch(r, "ornith_vocab_logits_q8_kernel", (s.vocab.rows + 3) / 4, 256, v);
                    struct { uint64_t scores, value, index; uint32_t count, first; }
                        top{s.logits, s.argmax, s.argmax + 4, s.vocab.rows, static_cast<uint32_t>(layout.ranks[r].vocab.begin)};
                    launch(r, "ornith_local_argmax_nrank", 1, 64, top);
                }
                stamp(r, 4 * hp.n_trunk + 1);
            }
            flush();
            book_token();
            float best = -std::numeric_limits<float>::infinity(); int32_t chosen = -1;
            // Canonical rank/range order gives lower-global-token tie breaking.
            for (size_t r : layout.rank_order) {
                const auto range = layout.ranks[r].vocab;
                if (!range.size()) continue;
                struct { float score; int32_t token; } top{};
                check(rt.download_from_vram(r, &top, ranks[r].argmax, sizeof(top)));
                if (!std::isfinite(top.score) || top.token < 0 || uint64_t(top.token) < range.begin || uint64_t(top.token) >= range.end)
                    throw std::runtime_error("generic Ornith: invalid vocabulary output");
                if (top.score > best) { best = top.score; chosen = top.token; }
            }
            if (chosen < 0) throw std::runtime_error("generic Ornith: empty vocabulary merge");
            return chosen;
        };
        auto generate = [&](const std::vector<int32_t>& ids, int32_t count, const OrnithGpuOptions::ServeTokenCb& callback) {
            if (ids.empty() || count < 0 || ids.size() + uint64_t(count) > context)
                throw std::runtime_error("generic Ornith: request exceeds context capacity");
            for (size_t r : layout.rank_order) for (const auto& l : ranks[r].layers) {
                if (l.state) zero(r, l.state, layout.ranks[r].state_bytes);
                if (l.conv) zero(r, l.conv, layout.ranks[r].conv_bytes);
            }
            flush(); // reset only after prior request completed; KV is overwritten causally
            int32_t next = -1;
            const double start = now_sec();
            for (uint32_t i = 0; i < ids.size(); ++i) next = step(ids[i], i);
            const double prefill = (now_sec() - start) * 1000;
            const double decode = now_sec();
            for (int32_t i = 0; i < count; ++i) {
                if (!callback(next, ids.size(), i + 1, prefill, (now_sec() - decode) * 1000)) break;
                if (i + 1 < count) next = step(next, static_cast<uint32_t>(ids.size()) + i);
            }
        };
        if (opt.serve_hook) {
            opt.serve_hook(generate, "Ornith generic raw TP=" + std::to_string(opt.tp));
            result.ok = true;
            return result;
        }
        int32_t next = -1;
        const double prefill_start = now_sec();
        for (uint32_t i = 0; i < prompt_ids.size(); ++i) next = step(prompt_ids[i], i);
        const double prefill_seconds = now_sec() - prefill_start;
        const double decode_start = now_sec();
        timing_on = opt.profile_stages;  // decode tokens only; prefill and serve requests are never stamped
        for (int i = 0; i < opt.decode_tokens; ++i) {
            generated.push_back(next);
            if (i + 1 < opt.decode_tokens) next = step(next, static_cast<uint32_t>(prompt_ids.size()) + i);
        }
        const double decode_seconds = now_sec() - decode_start;
        timing_on = false;
        if (opt.profile_stages) {
            if (!timed_tokens) std::printf("[RANK-TIMING note] no decode token was stamped (needs --n-predict >= 2)\n");
            else {
                const auto report = rank_timer.report("nrank-profile-stages", timed_tokens);
                std::printf("%s%s%s\n", ranktiming::to_table(report).c_str(), ranktiming::kLogTag, ranktiming::to_json(report).c_str());
                std::printf("[RANK-TIMING note] the decode tok/s of this run includes the stamp dispatches and read-backs\n");
            }
        }
        result.pp512_tps = prompt_ids.size() / prefill_seconds;
        result.tg128_raw_tps = opt.decode_tokens > 1 ? (opt.decode_tokens - 1) / decode_seconds : 0;
        result.parity_simple_text = tokenizer.decode(generated);
        std::printf("raw token IDs:");
        for (auto id : generated) std::printf(" %d", id);
        std::printf("\n%s\n", result.parity_simple_text.c_str());
        result.ok = true;
        std::printf("generic Ornith raw TP=%u: prefill=%zu generated=%zu; hardware token parity still requires an oracle comparison\n",
                    opt.tp, prompt_ids.size(), generated.size());
    } catch (const std::exception& e) {
        result.error = e.what(); result.ok = false;
    }
    return result;
}
} // namespace
