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

tsc
cp -r src/static dist
pkg . --target node18-linux-x64 --output builds/linux-build/dejpeg-linux

mv node_modules node_modules_linux

mkdir -p builds/linux-build/node_modules
cp -r node_modules_linux/onnxruntime-node builds/linux-build/node_modules
cp -r node_modules_linux/sharp builds/linux-build/node_modules

rm -rf builds/linux-build/node_modules/onnxruntime-node/bin/napi-v6/darwin
rm -rf builds/linux-build/node_modules/onnxruntime-node/bin/napi-v6/win32

rm -rf builds/linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/arm64
rm builds/linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime_providers_cuda.so
rm builds/linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*.bak

if command -v 7z >/dev/null 2>&1; then
  7z a -t7z -mx=9 builds/dejpeg-linux.7z ./builds/linux-build
  rm -rf builds/linux-build
else
  echo "NOTE: 7z is not installed, skipping compression"
fi

echo "build complete: dejpeg-linux"