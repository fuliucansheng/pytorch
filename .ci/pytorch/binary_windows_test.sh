#!/bin/bash
# shellcheck disable=SC1090
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2022

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
    export XPU_VERSION=2026.1
fi

# Only wheel packages use a wheel receipt; preserve the shared LibTorch path.
if [[ "${PACKAGE_TYPE:-}" == "wheel" && "$OS" != "windows-arm64" ]]; then
    verification_args=()
    if [[ "${WINDOWS_WHEEL_REQUIRE_SIGNED:-false}" == "true" ]]; then
        verification_args+=(--require-signed --authenticode)
    fi
    python "$PYTORCH_ROOT/.ci/wheel/windows/sign_wheel.py" check "$PYTORCH_FINAL_PACKAGE_DIR" "${verification_args[@]}"
fi

pushd "$PYTORCH_ROOT/.ci/pytorch/"

if [[ "$OS" == "windows-arm64" ]]; then
    ./windows/arm64/smoke_test.bat
else
    ./windows/internal/smoke_test.bat
fi

popd
