#!/usr/bin/env bash
# Copyright 2024-present the vsag project
# Licensed under the Apache License, Version 2.0
#
# 串行跑四组 Pyramid no_build_levels 配置(每组一个独立进程,
# 避免 glibc allocator 在同进程内的 pool 复用污染 RSS 对比)。
#
# 用法:
#   ./run_all.sh [<binary path>]
# 默认 binary 路径:同目录下的 pyramid_level_memory(由 CMake 构建产物)。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${1:-${SCRIPT_DIR}/pyramid_level_memory}"

if [[ ! -x "${BINARY}" ]]; then
    echo "binary not found or not executable: ${BINARY}" >&2
    echo "build it first, e.g.:" >&2
    echo "  cmake --build <build-dir> --target pyramid_level_memory" >&2
    exit 1
fi

CONFIGS=(only-day month-day year-month-day all)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "[run_all] launching config=${cfg}"
    echo "============================================================"
    "${BINARY}" --config "${cfg}"
done

echo "============================================================"
echo "[run_all] all configs finished. Compare 'delta (index-only)' across runs."
echo "============================================================"
