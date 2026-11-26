# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
cmake_minimum_required(VERSION 3.18)  # find_program(... REQUIRED), file(CONFIGURE)
project(VegaJet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_program(VEGAJET_CCACHE ccache)
if(VEGAJET_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${VEGAJET_CCACHE}")
endif()

option(VEGAJET_CPU_ONLY "Build CPU-only host tools; no ROCm/HIP headers, no /dev/kfd access" ON)
set(VEGAJET_ROCM_PATH "/opt/rocm" CACHE PATH "ROCm / TheRock installation prefix containing bin/hipcc and llvm/bin tools")
set(VEGAJET_GFX_ARCH "gfx906" CACHE STRING "Target AMDGPU ISA architecture (default: gfx906 Vega 20)")
set(VEGAJET_HSACO_DIR "${CMAKE_BINARY_DIR}/hsaco" CACHE PATH "Output/default directory for unbundled .hsaco code objects")

find_package(Threads REQUIRED)
find_package(OpenMP)

set(VJ_WARN -Wall -Wextra -Wpedantic)

# ---- gfx906 code-object gate table (generated from ops/kernels/hsaco_gates.txt) -----------------
set(VJ_GATE_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/hsaco_gates.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VJ_GATE_MANIFEST}")
file(STRINGS "${VJ_GATE_MANIFEST}" VJ_GATE_LINES)
set(VJ_GATE_INC "")
set(VJ_GATE_NAMES "")
foreach(line IN LISTS VJ_GATE_LINES)
    string(STRIP "${line}" line)
    if(line STREQUAL "" OR line MATCHES "^#")
        continue()
    endif()
    if(line MATCHES "^([A-Za-z0-9_]+)[ \t]+(stream|tiled)([ \t]+([0-9]+))?$")
        if("${CMAKE_MATCH_1}" IN_LIST VJ_GATE_NAMES)
            message(FATAL_ERROR "ops/kernels/hsaco_gates.txt: duplicate kernel: ${CMAKE_MATCH_1}")
        endif()
        list(APPEND VJ_GATE_NAMES "${CMAKE_MATCH_1}")
        set(vj_gate_waiver 0)
        if(CMAKE_MATCH_4)
            set(vj_gate_waiver "${CMAKE_MATCH_4}")
        endif()
        if(CMAKE_MATCH_2 STREQUAL "stream")
            set(vj_gate_class "KernelClass::Stream")
        else()
            set(vj_gate_class "KernelClass::Tiled")
        endif()
        string(APPEND VJ_GATE_INC "    {\"${CMAKE_MATCH_1}\", ${vj_gate_class}, ${vj_gate_waiver}u},\n")
    else()
        message(FATAL_ERROR "ops/kernels/hsaco_gates.txt: malformed line: ${line}")
    endif()
endforeach()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/hsaco_gates.inc" CONTENT "${VJ_GATE_INC}" @ONLY)

add_library(vegajet_core STATIC
    core/doctor.cpp
    core/host_preflight.cpp
    core/hsaco_gate.cpp
    core/hsaco_bundle.cpp
    core/gguf_arch_detect.cpp
    core/fr_spec.cpp
    core/spec_policy.cpp
    core/q4_dot8_repack.cpp
    core/moe_route_group.cpp
    core/prefill_planner.cpp
    core/tokenizer.cpp
    core/gguf_reader.cpp
    core/kfd_runtime.cpp
    core/quant_repacker.cpp
    core/quant_unpackers.cpp
    core/sharder.cpp
    core/topology_planner.cpp
    core/hw_profile.cpp
    core/kv_cache.cpp
    core/dflash_host.cpp
    core/ssm_state.cpp
    core/spec_engine.cpp
    core/spec_tree.cpp
    core/grid_planner.cpp
    core/launch_planner.cpp
    core/sampler.cpp
    core/server.cpp
    core/bpe_tokenizer.cpp
    core/hsaco_isa_scan.cpp
    core/vram_planner.cpp
    core/moe_lru_cache.cpp
    core/moe_offload.cpp
    core/suffix_trie.cpp
    core/spec_cascade.cpp
    core/paged_kv.cpp
    core/ple_pregather.cpp
    core/doctor_rules.cpp
    core/kv_quant_oracle.cpp
    core/evict_controller.cpp
    core/tp3_plan.cpp
    core/async_table_prefetch.cpp
    core/warm_bench.cpp
    core/tool_call_parser.cpp
    core/request_scheduler.cpp
    core/gpu_select.cpp
    core/engine_factory.cpp
    ops/op_desc.cpp
    ops/layer_plan.cpp
)
target_link_libraries(vegajet_core PUBLIC Threads::Threads)
target_include_directories(vegajet_core PRIVATE "${CMAKE_BINARY_DIR}/generated")
# `vegajet doctor` checks the ROCm toolchain this build was configured with (core/host_preflight).
target_compile_definitions(vegajet_core PRIVATE VEGAJET_BUILD_ROCM_PATH="${VEGAJET_ROCM_PATH}")

# Build-time gate tools (also usable by hand: vj_hsaco_gate / vj_isa_scan <file.hsaco>...)
add_executable(vj_hsaco_gate tools/hsaco_gate_main.cpp)
target_include_directories(vj_hsaco_gate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vj_hsaco_gate PRIVATE vegajet_core Threads::Threads)

add_executable(vj_isa_scan tools/vj_isa_scan_main.cpp)
target_include_directories(vj_isa_scan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vj_isa_scan PRIVATE vegajet_core Threads::Threads)

add_executable(vj_mock_serve tools/vj_mock_serve_main.cpp)
target_include_directories(vj_mock_serve PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vj_mock_serve PRIVATE vegajet_core Threads::Threads)
target_include_directories(vegajet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vegajet_core PRIVATE ${VJ_WARN})
if(OpenMP_CXX_FOUND)
    target_link_libraries(vegajet_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(VEGAJET_CPU_ONLY)
    target_compile_definitions(vegajet_core PUBLIC VEGAJET_CPU_ONLY=1)
else()
    target_compile_definitions(vegajet_core PUBLIC
        VEGAJET_DEFAULT_HSACO_DIR="${VEGAJET_HSACO_DIR}"
    )
endif()

# ---- GPU code objects (.hsaco) when VEGAJET_CPU_ONLY=OFF ---------------------------------------
if(NOT VEGAJET_CPU_ONLY)
    find_program(VEGAJET_HIPCC hipcc HINTS "${VEGAJET_ROCM_PATH}/bin" REQUIRED)
    find_program(VEGAJET_OFFLOAD_BUNDLER clang-offload-bundler HINTS "${VEGAJET_ROCM_PATH}/llvm/bin" "${VEGAJET_ROCM_PATH}/lib/llvm/bin" "${VEGAJET_ROCM_PATH}/bin" REQUIRED)
    find_program(VEGAJET_LLVM_OBJDUMP llvm-objdump HINTS "${VEGAJET_ROCM_PATH}/llvm/bin" "${VEGAJET_ROCM_PATH}/lib/llvm/bin" "${VEGAJET_ROCM_PATH}/bin" REQUIRED)

    file(MAKE_DIRECTORY "${VEGAJET_HSACO_DIR}")

    set(VEGAJET_HSACO_COMMON_DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/core/ggml_iq_tables.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_moe_dot8.hiph"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_moe_down_grouped.hiph"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_ornith_super_desc.hpp"
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_ornith_common.hiph")
        list(APPEND VEGAJET_HSACO_COMMON_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_ornith_common.hiph")
    endif()

    set(VEGAJET_HSACO_TARGETS "")
    function(vj_add_hsaco name src_file)
        set(bundle_out "${VEGAJET_HSACO_DIR}/${name}.bundle")
        set(hsaco_out  "${VEGAJET_HSACO_DIR}/${name}.hsaco")
        set(extra_flags ${ARGN})
        add_custom_command(
            OUTPUT "${hsaco_out}"
            COMMAND "${VEGAJET_HIPCC}" --genco --offload-arch=${VEGAJET_GFX_ARCH} -O3
                    -I "${CMAKE_CURRENT_SOURCE_DIR}"
                    ${extra_flags}
                    -o "${bundle_out}" "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
            COMMAND "${VEGAJET_OFFLOAD_BUNDLER}" --type=o
                    "--targets=hipv4-amdgcn-amd-amdhsa--${VEGAJET_GFX_ARCH}"
                    "--input=${bundle_out}" "--output=${hsaco_out}" --unbundle
            COMMAND vj_hsaco_gate "${hsaco_out}"
            COMMAND vj_isa_scan "${hsaco_out}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" ${VEGAJET_HSACO_COMMON_DEPS} vj_hsaco_gate vj_isa_scan
            COMMENT "Building gated (0 B spill, VGPR, code size, ISA scan) ${VEGAJET_GFX_ARCH} code object ${name}.hsaco"
            VERBATIM
        )
        add_custom_target(hsaco_${name} ALL DEPENDS "${hsaco_out}")
        add_custom_target(${name}_hsaco DEPENDS "${hsaco_out}")
        list(APPEND VEGAJET_HSACO_TARGETS hsaco_${name})
        set(VEGAJET_HSACO_TARGETS "${VEGAJET_HSACO_TARGETS}" PARENT_SCOPE)
    endfunction()

    vj_add_hsaco(vj_kfd_base           ops/kernels/vj_kfd_base.hip)
    vj_add_hsaco(vj_quant_ops          ops/kernels/vj_quant_ops.hiph -Wno-pragma-once-outside-header -x hip)
    vj_add_hsaco(vj_kv_flash_ops       ops/kernels/vj_kv_flash_ops.hiph -Wno-pragma-once-outside-header -x hip)
    vj_add_hsaco(vj_ssm_neumann_ops    ops/kernels/vj_ssm_neumann_ops.hiph -Wno-pragma-once-outside-header -x hip)
    vj_add_hsaco(vegajet_megakernel_vm ops/kernels/vegajet_megakernel_vm.hip)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/universal_draft/ops/vj_universal_vm.hip")
        vj_add_hsaco(vj_universal_vm   universal_draft/ops/vj_universal_vm.hip)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/vj_ornith_ar.hip")
        vj_add_hsaco(vj_ornith_ar      ops/kernels/vj_ornith_ar.hip)
        vj_add_hsaco(vj_ornith_gemm    ops/kernels/vj_ornith_gemm.hip)
        vj_add_hsaco(vj_ornith_attn    ops/kernels/vj_ornith_attn.hip)
        vj_add_hsaco(vj_ornith_moe     ops/kernels/vj_ornith_moe.hip)
        vj_add_hsaco(vj_ornith_moe_m1  ops/kernels/vj_ornith_moe_m1.hip)
        vj_add_hsaco(vj_moe_asym       ops/kernels/vj_moe_asym.hip)
        vj_add_hsaco(vj_ornith_vocab   ops/kernels/vj_ornith_vocab.hip)
        vj_add_hsaco(vj_spec_flash     ops/kernels/vj_spec_flash.hip)
        vj_add_hsaco(vj_spec_gdn       ops/kernels/vj_spec_gdn.hip)
        vj_add_hsaco(vj_moe_dot8       ops/kernels/vj_moe_dot8.hip)
        vj_add_hsaco(vj_prefill_fa2    ops/kernels/vj_prefill_fa2.hip)
        vj_add_hsaco(vj_prefill_gemm   ops/kernels/vj_prefill_gemm.hip)
        vj_add_hsaco(vj_prefill_dot8   ops/kernels/vj_prefill_dot8.hip)
        vj_add_hsaco(vj_prefill_gdn    ops/kernels/vj_prefill_gdn.hip)
        vj_add_hsaco(vj_moe_down_grouped ops/kernels/vj_moe_down_grouped.hip)
        vj_add_hsaco(vj_dflash2        ops/kernels/vj_dflash2.hip)
        vj_add_hsaco(vj_ornith_super   ops/kernels/vj_ornith_super.hip)
        # v1.2 opt-in variants: same source, own symbol suffix (see VJ_KNAME), own gate lines.
        # VEGAJET_V12_BARRIER_VOL=1 binds *_vol (buffer_wbinvl1_vol stage barriers).
        vj_add_hsaco(vj_ornith_super_vol ops/kernels/vj_ornith_super.hip -DVJ_V12_WBINVL1_VOL -DVJ_KERNEL_SUFFIX=_vol)
        # VEGAJET_V12_IQ2S_REG=1 binds *_iq2sreg (IQ2_S Gate+Up decodes from the L1-resident 4 KiB
        # grid table, no LDS copy -> 8,896 B -> 4,800 B LDS per WG).
        vj_add_hsaco(vj_ornith_super_iq2sreg ops/kernels/vj_ornith_super.hip -DVJ_V12_IQ2S_REG -DVJ_KERNEL_SUFFIX=_iq2sreg)
    endif()
    add_custom_target(vegajet_hsacos DEPENDS ${VEGAJET_HSACO_TARGETS})
endif()

set(VEGAJET_MODEL_SOURCES
    models/registry.cpp
    models/builtin_models.cpp
    models/qwen4exp/qwen4exp_model.cpp
    models/ornith/ornith_loader.cpp
    models/ornith/ornith_runner.cpp
    models/ornith/ornith_gpu.cpp
    models/model_blueprints.cpp
)

add_library(vegajet_models STATIC ${VEGAJET_MODEL_SOURCES})
target_sources(vegajet_models PRIVATE agent/MODEL_TEMPLATE/declarative_model.cpp)
target_include_directories(vegajet_models PUBLIC agent/MODEL_TEMPLATE)
target_link_libraries(vegajet_models PUBLIC vegajet_core)
target_compile_options(vegajet_models PRIVATE ${VJ_WARN})

# `vegajet serve` (+ the CLI-wide --gpu option) as a library so tests can link it.
add_library(vegajet_cli STATIC cli/serve_cmd.cpp)
target_link_libraries(vegajet_cli PUBLIC vegajet_core Threads::Threads)
target_compile_options(vegajet_cli PRIVATE ${VJ_WARN})

add_executable(vegajet
    cli/main.cpp
    cli/inspect.cpp
    cli/bench.cpp
    cli/soak.cpp
)
target_link_libraries(vegajet PRIVATE vegajet_cli vegajet_models Threads::Threads)
target_compile_options(vegajet PRIVATE ${VJ_WARN})
if(NOT VEGAJET_CPU_ONLY)
    add_dependencies(vegajet vegajet_hsacos)
endif()

# ---- tests -------------------------------------------------------------------------------------
enable_testing()

function(vj_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE vegajet_models Threads::Threads)
    target_compile_options(${name} PRIVATE ${VJ_WARN})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

vj_test(test_tokenizer)
vj_test(test_gguf_reader)
# v1.2 parser hardening (AUDIT-parsers.md): malformed GGUF / tokenizer / template inputs.
vj_test(test_parser_gguf_overflow)
vj_test(test_parser_gguf_ne)
vj_test(test_parser_gguf_tensor_count)
vj_test(test_parser_ornith_nextn)
vj_test(test_parser_ornith_div0)
vj_test(test_parser_tok_count)
vj_test(test_parser_tok_nested)
vj_test(test_quant_repacker)
# Keep synthetic fixture rounding identical to the golden inputs on FMA-capable hosts.
target_compile_options(test_quant_repacker PRIVATE -ffp-contract=off)
# ---- Universal unpacker registry + `vegajet plan --gpus N --vram G` rig estimator (core_quant_registry_plan) ----
target_sources(vegajet_core PRIVATE core/quant_registry.cpp core/rig_planner.cpp)
target_sources(vegajet PRIVATE cli/plan_rig.cpp)
vj_test(test_quant_registry_plan)
vj_test(test_kfd_arena_tiers)
vj_test(test_aql_ring_deadlock)
vj_test(test_quant_unpackers)
vj_test(test_ornith_loader)
vj_test(test_sharder)
vj_test(test_topology_planner)
vj_test(test_grid_planner)
vj_test(test_hsaco_gate)
# v1.2 C1/C3: device fault-record decoding + token-id bounds guard (host side).
vj_test(test_device_guard)
vj_test(test_v12_dispatch)
vj_test(test_iq2s_perm4)
# ---- Pre-tokenizer families + SPM, Jinja-subset chat templates, full sampler stack (core_tokenizer_samplers) ----
target_sources(vegajet_core PRIVATE core/pretok_families.cpp core/chat_template.cpp core/sampler_stack.cpp)
vj_test(test_tokenizer_samplers)
# Its 248K-vocabulary latency gate (< 150 us per token) must not share the CPU with other tests.
set_tests_properties(test_tokenizer_samplers PROPERTIES RUN_SERIAL TRUE)
vj_test(test_spec_engine)
vj_test(test_spec_tree)
target_link_libraries(test_spec_tree PRIVATE vegajet_models)
vj_test(test_fr_spec)
vj_test(test_spec_policy)
vj_test(test_spec_governor)
vj_test(test_spec_flash_ref)
vj_test(test_hsaco_bundle)
vj_test(test_dflash_host)
vj_test(test_spec_gdn_ref)
vj_test(test_gdn_recur_colthread_ref)
vj_test(test_gemv_q4a8_mn_ref)
vj_test(test_q4_dot8)
# ---- Radix prefix cache (KV blocks + recurrent-state snapshots) and B=1..8 continuous batching (serving_prefix_batching) ----
target_sources(vegajet_core PRIVATE core/prefix_cache.cpp core/cont_batch.cpp)
vj_test(test_prefix_batching)
vj_test(test_decode_vocab_mubuf_argmax)
vj_test(test_decode_vocab_ilp_argmax)
vj_test(test_dense_persist_tiles)
vj_test(test_moe_route_group)
vj_test(test_prefill_planner)
vj_test(test_prefill_ref)
vj_test(test_prefill_dot8_kld)
vj_test(test_audit_invariants)
vj_test(test_tcc_anti_camping)
vj_test(test_grid_planner_splitk)
vj_test(test_asymmetric_sharder)
vj_test(test_asym_experts)
vj_test(test_opgraph_multi_arch)
vj_test(test_sampler)
vj_test(test_host_preflight)
vj_test(test_server_sse)
# v1.2 serving levers: greedy-only 400s, bounded async SSE output queue (byte-identical, ordered, disconnect-safe).
vj_test(test_serve_v12)
vj_test(test_declarative_template)
vj_test(test_hsaco_isa_scan)
vj_test(test_vram_planner)
vj_test(test_moe_lru_cache)
vj_test(test_moe_offload)
vj_test(test_suffix_trie)
vj_test(test_spec_cascade)
vj_test(test_fr_spec_lossless)
vj_test(test_fwht_kld_oracle)
vj_test(test_gdn_verify_oracle)
vj_test(test_bpe_tokenizer)
vj_test(test_paged_kv)
vj_test(test_ple_pregather)
vj_test(test_doctor_mock_sysfs)
vj_test(test_platform_doctor)
vj_test(test_kv_quant_oracle)
vj_test(test_evict_controller)
vj_test(test_tp3_plan)
# ---- GBNF / JSON Schema constrained decoding, multi-format tool-call parser, /v1/messages, swarm router (serving_grammar_api_swarm) ----
target_sources(vegajet_core PRIVATE core/grammar.cpp core/tool_parser.cpp core/anthropic_shim.cpp core/swarm_router.cpp)
vj_test(test_grammar_api_swarm)
vj_test(test_async_table_prefetch)
vj_test(test_model_blueprints)
vj_test(test_tool_call_parser)
vj_test(test_server_tools)
vj_test(test_request_scheduler)
vj_test(test_gpu_select)
vj_test(test_serve_cli)
target_link_libraries(test_serve_cli PRIVATE vegajet_cli)
target_link_libraries(test_gpu_select PRIVATE vegajet_cli)
# ---- Prometheus /metrics + JSONL request log, `vegajet top`, `vegajet chat` (cli_observability_chat) ----
target_sources(vegajet_core PRIVATE core/metrics.cpp)
target_sources(vegajet_cli PRIVATE cli/http_mini.cpp cli/tui_top.cpp cli/chat_repl.cpp)
vj_test(test_observability_chat)
target_link_libraries(test_observability_chat PRIVATE vegajet_cli)
# Writes a miniature qwen35moe GGUF (with a tokenizer) for the CLI end-to-end tests.
add_executable(vj_mini_gguf tests/vj_mini_gguf_main.cpp)
target_link_libraries(vj_mini_gguf PRIVATE vegajet_core)
target_compile_options(vj_mini_gguf PRIVATE ${VJ_WARN})
vj_test(test_warm_bench_protocol)

add_executable(vegajet_layer_test tests/vegajet_layer_test.cpp)
target_link_libraries(vegajet_layer_test PRIVATE vegajet_models Threads::Threads)
target_compile_options(vegajet_layer_test PRIVATE ${VJ_WARN})
if(NOT VEGAJET_CPU_ONLY)
    add_dependencies(vegajet_layer_test vegajet_hsacos)
endif()
add_test(NAME vegajet_layer_test COMMAND vegajet_layer_test --stage all --m 1,16,128)
set_tests_properties(vegajet_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
target_compile_definitions(test_opgraph_multi_arch PRIVATE
    VJ_LAYER_PLAN_SRC="${CMAKE_SOURCE_DIR}/ops/layer_plan.cpp")
target_include_directories(test_hsaco_isa_scan PRIVATE "${CMAKE_BINARY_DIR}/generated")
# The CPU model replays the device float chain: fused multiply-adds must stay exactly where the code spells them.
set_source_files_properties(tests/test_moe_route_group.cpp core/moe_route_group.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME test_v12_fix_contracts
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_v12_fix_contracts.py)
    add_test(NAME test_tools
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_tools*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(NAME test_swarm_router
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_swarm_router*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_swarm_router PROPERTIES ENVIRONMENT "VJ_MOCK_SERVE=$<TARGET_FILE:vj_mock_serve>")
    add_test(NAME test_regression_gate
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_regression_gate*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(NAME test_cli_e2e
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_cli_e2e*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_cli_e2e PROPERTIES
             ENVIRONMENT "VEGAJET_BIN=$<TARGET_FILE:vegajet>;VJ_MINI_GGUF=$<TARGET_FILE:vj_mini_gguf>")
    add_test(NAME test_cli_reference
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_cli_reference*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_cli_reference PROPERTIES ENVIRONMENT "VEGAJET_BIN=$<TARGET_FILE:vegajet>")
    add_test(NAME api_stress_test
             COMMAND ${Python3_EXECUTABLE} tools/api_stress_test.py --spawn $<TARGET_FILE:vj_mock_serve>
                     --clients 24 --rounds 3 --spawn-arg=--max-queue --spawn-arg=2 --expect-backpressure
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    # JSON-only diff/regression gate for two `vegajet bench --json-out` scorecards (stdlib, no log parsing).
    add_test(NAME test_bench_compare_selftest
             COMMAND ${Python3_EXECUTABLE} tools/bench_compare.py --self-test
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_ab_matrix.sh")
    add_test(NAME test_guard_latch COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_guard_latch.sh")
endif()
add_test(NAME test_contrib_hardware
         COMMAND bash tests/test_contrib_hardware.sh
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# Offline 3x MI50 bring-up planner (tools/vj_tp3_preflight.py): reads GGUF headers only, never a device.
if(Python3_FOUND)
    add_test(NAME test_tp3_preflight
             COMMAND ${Python3_EXECUTABLE} -B -m unittest discover -s tests -p "test_tp3_preflight*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
add_test(NAME test_vj_dev_selftest
         COMMAND bash tools/vj_dev.sh --self-test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vj_bench_warm_selftest
         COMMAND bash tools/vj_bench_warm.sh --self-test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vj_commit_gate_selftest
         COMMAND bash tools/vj_commit_gate.sh --self-test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_apply_handoff_selftest
         COMMAND bash tools/apply_handoff.sh --self-test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ---- Alpha portability (core/portability): relaxed rig-specific doctor checks, device-access probe, fit check ----
target_sources(vegajet_core PRIVATE core/portability.cpp)
vj_test(test_portability)

add_test(NAME vegajet_cli_help COMMAND vegajet --help)
# Use an empty synthetic host; a CPU-only build must never probe the test runner's GPUs.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/empty-doctor-root")
add_test(NAME vegajet_doctor_dry_run COMMAND vegajet doctor --os --dry-run --root "${CMAKE_CURRENT_BINARY_DIR}/empty-doctor-root")
set_tests_properties(vegajet_doctor_dry_run PROPERTIES WILL_FAIL TRUE)

# ---- `vegajet bench --json-out` (core/bench_json, schema vegajet.bench.v1) ----------------------
target_sources(vegajet_core PRIVATE core/bench_json.cpp)
# -DVEGAJET_GIT_COMMIT=<sha> overrides (builds from a `git archive` export have no repository to ask).
set(VEGAJET_GIT_COMMIT "" CACHE STRING "commit recorded in bench JSON (default: git rev-parse at configure time)")
set(VJ_GIT_COMMIT "${VEGAJET_GIT_COMMIT}")
if(NOT VJ_GIT_COMMIT)
    execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE VJ_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT VJ_GIT_COMMIT)
    set(VJ_GIT_COMMIT "unknown")
endif()
# Captured at configure time: re-run cmake after a commit to refresh it.
target_compile_definitions(vegajet PRIVATE VEGAJET_GIT_COMMIT="${VJ_GIT_COMMIT}")
vj_test(test_bench_json)
if(Python3_FOUND)
    add_test(NAME test_bench_json_roundtrip
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_bench_json_roundtrip*.py"
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_bench_json_roundtrip PROPERTIES ENVIRONMENT "BENCH_JSON_EMIT=$<TARGET_FILE:test_bench_json>")
endif()

# ---- Phase 2C: Codex 3-kernel zero-spill universal engine (universal_draft/) --------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/universal_draft/ops/vj_vm_cpu.cpp")
    add_library(vegajet_universal STATIC
        universal_draft/core/emit_bytecode.cpp
        universal_draft/core/fusion_pass.cpp
        universal_draft/core/onboard_shard.cpp
        universal_draft/core/tiny_execution.cpp
        universal_draft/core/universal_ir.cpp
        universal_draft/core/universal_quant.cpp
        universal_draft/core/workspace_packer.cpp
        universal_draft/ops/vj_vm_cpu.cpp
    )
    target_include_directories(vegajet_universal PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/universal_draft"
        "${CMAKE_CURRENT_SOURCE_DIR}/universal_draft/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/universal_draft/ops"
    )
    target_link_libraries(vegajet_universal PUBLIC vegajet_core)
    target_compile_options(vegajet_universal PRIVATE ${VJ_WARN} -ffp-contract=off)
    foreach(_utest IN ITEMS universal_compiler universal_quant universal_vm plug_and_play_e2e)
        add_executable(test_${_utest} "universal_draft/tests/test_${_utest}.cpp")
        target_link_libraries(test_${_utest} PRIVATE vegajet_universal)
        target_compile_options(test_${_utest} PRIVATE ${VJ_WARN} -ffp-contract=off)
        add_test(NAME test_${_utest} COMMAND test_${_utest})
    endforeach()
    if(NOT VEGAJET_CPU_ONLY AND Python3_FOUND)
        add_test(NAME test_vj_onboard_audit
                 COMMAND ${Python3_EXECUTABLE} -B universal_draft/tools/vj_onboard_audit.py
                         --self-test --reader tools/hsaco_audit.py
                         --hsaco "${VEGAJET_HSACO_DIR}/vj_universal_vm.hsaco"
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        set_tests_properties(test_vj_onboard_audit PROPERTIES
                             ENVIRONMENT "PATH=${VEGAJET_ROCM_PATH}/lib/llvm/bin:$ENV{PATH}")
    endif()
    if(NOT VEGAJET_CPU_ONLY)
        add_executable(test_universal_vm_gpu tests/test_universal_vm_gpu.cpp)
        target_link_libraries(test_universal_vm_gpu PRIVATE vegajet_universal)
        target_compile_options(test_universal_vm_gpu PRIVATE ${VJ_WARN})
        add_test(NAME test_universal_vm_gpu
                 COMMAND test_universal_vm_gpu "${VEGAJET_HSACO_DIR}/vj_universal_vm.hsaco" "${VEGAJET_HSACO_DIR}/vj_kfd_base.hsaco")
        set_tests_properties(test_universal_vm_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

# ---- Sealed alpha ISA forge deliverables (CPU contract tests) -----------------------------------
foreach(_isa_test IN ITEMS flash_decode_splitkv gdn_treewy_verify moe_lru_refill_dot8 ar_tp3_alltoall)
    add_executable(test_${_isa_test} "tests/test_${_isa_test}.cpp")
    target_include_directories(test_${_isa_test} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_${_isa_test} PRIVATE Threads::Threads)
    target_compile_options(test_${_isa_test} PRIVATE ${VJ_WARN} -UNDEBUG)
    add_test(NAME test_${_isa_test} COMMAND test_${_isa_test})
endforeach()
