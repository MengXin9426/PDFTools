#!/usr/bin/env bash
# 禁用全局 pytest 插件（如 ROS），避免缺依赖导致收集失败。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
echo "[pdftools] PYTEST_DISABLE_PLUGIN_AUTOLOAD=$PYTEST_DISABLE_PLUGIN_AUTOLOAD" >&2
echo "[pdftools] python -m pytest $*" >&2
exec python -m pytest "$@"
