#!/bin/bash

set -e
cd "$(dirname "$0")"
cd ..

if [ -d node_modules ]; then
  echo "node_modules directory already exists. quitting."
  exit 1
fi

rm -rf dist

mv node_modules_windows node_modules

tsc
cp -r src/static dist
pkg . --target node18-win-x64 --output builds/win-build/dejpeg-win.exe

mv node_modules node_modules_windows

mkdir -p builds/win-build/node_modules
cp -r node_modules_windows/onnxruntime-node builds/win-build/node_modules
cp -r node_modules_windows/sharp builds/win-build/node_modules

rm -rf builds/win-build/node_modules/onnxruntime-node/bin/napi-v6/darwin
rm -rf builds/win-build/node_modules/onnxruntime-node/bin/napi-v6/linux

rm -rf builds/win-build/node_modules/onnxruntime-node/bin/napi-v6/win32/arm64

7z a -t7z -mx=9 builds/dejpeg-win.7z ./builds/win-build
rm -rf builds/win-build

echo "build complete: dejpeg-win.7z"