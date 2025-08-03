#!/bin/bash

set -e
cd "$(dirname "$0")"
cd ..

if [ -d node_modules ]; then
  echo "node_modules directory already exists. quitting."
  exit 1
fi

mv node_modules_linux node_modules

tsc && cp -r src/static dist
pkg . --target node18-linux-x64 --output linux-build/dejpeg-linux

mv node_modules node_modules_linux

mkdir -p linux-build/node_modules
cp -r node_modules_linux/onnxruntime-node linux-build/node_modules
cp -r node_modules_linux/sharp linux-build/node_modules

# removing unnecessary binaries
rm -rf linux-build/node_modules/onnxruntime-node/bin/napi-v6/darwin
rm -rf linux-build/node_modules/onnxruntime-node/bin/napi-v6/win32

# stripping unnecessary files
rm -rf linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/arm64
rm linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime_providers_cuda.so
rm linux-build/node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*.bak

7z a -t7z -mx=9 ./dejpeg-linux.7z ./linux-build
# clean build directory
rm -rf linux-build

echo "build complete: dejpeg-linux"