# dejpeg-desktop

beta version of DeJPEG for desktops (an ONNX gui that ISN'T electron)

prebuilt binaries available under `Releases` or build below

preq:

nodejs 18 or newer

```bash
npm install express cors multer pkg sharp@0.32.6 nodemon @types/express @types/cors @types/multer @types/node onnxruntime-node@latest
```

and

```
npm install typescript -g
```

then

```bash
npm run build && npm start
```

<!-- then `npm run build` or manually copy `src/static` to `dist` if on windows -->

# building

## Windows system

`pkg . --target node18-win-x64 --output dejpeg-windows.exe`

# Linux system

rename `node_modules` to `node_modules_linux` (or if you have a windows version of `node_modules`, it should be `node_modules_windows`) then run `npm run build-linux` (or `npm run build-win` for windows)

this will compile and zip the build (if 7z exists)

more info below why

it is possible to cross-platform build, however since i dont use windows i needed to use a vm to setup the build env, copy windows' `node_modules` to my linux environment and rename/replace linux's `node_modules` folder with the windows version, then run a windows build command. so now I dont need to use a windows vm when releasing a new version yay, this is what `scripts/build-(platform).sh` is for. I don't use Windows therefore no batch/powershell versions for these
