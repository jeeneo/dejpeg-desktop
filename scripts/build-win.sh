#!/bin/bash

set -e
cd "$(dirname "$0")"
cd ..

if [ -d node_modules ]; then
  echo "node_modules directory already exists. quitting."
  exit 1
fi

mv node_modules_windows node_modules

tsc && cp -r src/static dist
mkdir -p win-build
pkg . --target node18-win-x64 --output win-build/dejpeg-win.exe

mv node_modules node_modules_windows

mkdir -p win-build/node_modules
cp -r node_modules_windows/onnxruntime-node win-build/node_modules
cp -r node_modules_windows/sharp win-build/node_modules

rm -rf win-build/node_modules/onnxruntime-node/bin/napi-v6/darwin
rm -rf win-build/node_modules/onnxruntime-node/bin/napi-v6/linux

rm -rf win-build/node_modules/onnxruntime-node/bin/napi-v6/win32/arm64

7z a -t7z -mx=9 ./dejpeg-win.7z ./win-build/*
# clean build directory
rm -rf win-build

echo "build complete: dejpeg-win.7z"