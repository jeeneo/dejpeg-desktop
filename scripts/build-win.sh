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

[ -d node_modules/onnxruntime-node/bin/napi-v6/darwin ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/darwin
[ -d node_modules/onnxruntime-node/bin/napi-v6/linux ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/linux
[ -d node_modules/onnxruntime-node/bin/napi-v6/win32/arm64 ] && rm -rf node_modules/onnxruntime-node/bin/napi-v6/win32/arm64

tsc
cp -r src/static dist
pkg . --target node18-win-x64 --output builds/dejpeg-win.exe

mv node_modules node_modules_windows

if command -v 7z >/dev/null 2>&1; then
  7z a -t7z -mx=9 builds/dejpeg-win.7z ./builds/dejpeg-win.exe
  rm -rf builds/dejpeg-win.exe
else
  echo "NOTE: 7z is not installed, skipping compression"
fi

echo "build complete: dejpeg-win.7z"