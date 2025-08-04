#!/bin/bash

set -e
cd "$(dirname "$0")"
cd ..

if [ -d node_modules ]; then
  echo "node_modules directory already exists. quitting."
  exit 1
fi

rm -rf dist

mv node_modules_linux node_modules

[ -d node_modules/onnxruntime-node/bin/napi-v6/darwin ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/darwin
[ -d node_modules/onnxruntime-node/bin/napi-v6/win32 ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/win32
[ -d node_modules/onnxruntime-node/bin/napi-v6/linux/arm64 ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/linux/arm64
[ -f node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime_providers_cuda.so ] && rm node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime_providers_cuda.so # comment out to keep CUDA
[ -n "$(ls node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*.bak 2>/dev/null)" ] && rm node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*.bak

tsc
cp -r src/static dist
pkg . --target node18-linux-x64 --output builds/dejpeg-linux

mv node_modules node_modules_linux

if command -v 7z >/dev/null 2>&1; then
  7z a -t7z -mx=9 builds/dejpeg-linux.7z ./builds/dejpeg-linux
  rm -rf builds/dejpeg-linux
else
  echo "NOTE: 7z is not installed, skipping compression"
fi

echo "build complete: dejpeg-linux"