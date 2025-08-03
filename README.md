# dejpeg-desktop

beta version of DeJPEG for desktops (yet another ONNX gui)

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

for windows, manually copy `src/static` to `dist`

### for building into single executable:

linux:

```bash
pkg . --target node18-linux-x64 --output dejpeg-linux
```

windows

```batch
pkg . --target node18-win-x64 --output dejpeg-windows
```

(it is possible to cross-platform build, however since i dont use windows i needed to use a vm to setup the build env, copy windows' `node_modules` to my linux environment and rename/replace linux's `node_modules` folder with the windows version, then run a windows build command. so now I dont need to use a windows vm when releasing a new version yay)