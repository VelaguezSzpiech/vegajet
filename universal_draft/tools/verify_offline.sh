#!/usr/bin/env bash
# Run from the draft or its parent. Supply an empty, writable build directory.
set -eu -o pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
build=$(realpath "${1:?provide offline build directory}")
mkdir -p "$build/tmp"; export TMPDIR="$build/tmp"
base=${2:?provide immutable baseline directory}
llvm=${LLVM_BIN:-$(dirname "$(command -v clang-offload-bundler)")}
export PATH="$llvm:$PATH"
flags=(-std=c++17 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -I"$src" -I"$src/core" -I"$src/ops" -I"$base")
for mode in opt san; do
  opts=(-O3); if [[ $mode == san ]]; then opts=(-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -no-pie); fi
  mkdir -p "$build/$mode"; objects=()
  for source in "$base/core/gguf_reader.cpp" "$src"/core/*.cpp "$src"/ops/*.cpp; do
    object="$build/$mode/$(basename "${source%.cpp}").o"; "${CXX:-g++}" "${flags[@]}" "${opts[@]}" -c "$source" -o "$object"; objects+=("$object")
  done
  for test in universal_compiler universal_quant universal_vm plug_and_play_e2e; do
    exe="$build/$mode/test_$test"; "${CXX:-g++}" "${flags[@]}" "${opts[@]}" "$src/tests/test_$test.cpp" "${objects[@]}" -o "$exe"
    (cd "$build/$mode" && "$exe") | tee "$exe.log"
  done
done
"${HIPCC:-hipcc}" --genco --offload-arch=gfx906 -std=c++17 -Wall -Wextra -Wpedantic -Werror -O3 "$src/ops/vj_universal_vm.hip" -o "$build/vm.bundle"
clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx906 --input="$build/vm.bundle" --output="$build/vm.hsaco" --unbundle
llvm-objdump -d --mcpu=gfx906 "$build/vm.hsaco" > "$build/vm.isa"
python3 -B "$src/tools/vj_onboard_audit.py" --self-test --reader "$base/tools/hsaco_audit.py" --hsaco "$build/vm.hsaco" | tee "$build/isa.log"
