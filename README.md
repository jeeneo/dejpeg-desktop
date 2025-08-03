# dejpeg-desktop

beta version of DeJPEG for desktops (an ONNX gui that ISN'T electron)

prebuilt binaries available under `Releases` or build below

preq:

`nodejs` v22

```bash
npm install express cors multer pkg sharp@0.32.6
```

```bash
npm install --save-dev nodemon @types/express @types/cors @types/multer @types/node
```

then

```bash
npm run build && npm start
```

then `npm run build` or manually copy `src/static` to `dist` if on windows

## Windows

for windows, run `pkg . --target node18-win-x64 --output dejpeg-windows`, and copy both:

```
node_modules/onnxruntime-node
node_modules/sharp
```

to wherever you run alongside the exe

# Linux

rename `node_modules` to `node_modules_linux` (or if you have a windows version of `node_modules`, it should be `node_modules_windows`) then run `npm run build-linux` (or `npm run build-win` for windows)

this will copy, compile, and zip the build (if 7z exists)

more info below why

it is possible to cross-platform build, however since i dont use windows i needed to use a vm to setup the build env, copy windows' `node_modules` to my linux environment and rename/replace linux's `node_modules` folder with the windows version, then run a windows build command. so now I dont need to use a windows vm when releasing a new version yay, this is what `scripts/build-(platform).sh` is for. I don't use Windows therefore no batch/powershell versions for these