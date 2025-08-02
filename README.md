# dejpeg-desktop
another onnx gui in NodeJS

beta version of DeJPEG for desktops (yet another ONNX gui)

(no windows builds yet)

preq:
```
npm install express cors multer sharp@0.32.6
```

```
npm install --save-dev nodemon @types/express @types/cors @types/multer @types/node
```

then

```
npm run build && npm start
```

for building into single executable:

linux:

```
pkg . --target node18-linux-x64 --output dejpeg-node
```