# dejpeg-desktop
another onnx gui in NodeJS

beta version of a desktop DeJPEG, only fbcnn_color tested, don't file issues right away

no windows builds, only linux for now

preq:
`npm install express cors multer sharp@0.32.6`
`npm install --save-dev nodemon @types/express @types/cors @types/multer @types/node`

copy `src/static` to `dist` before building (i'm not nodejs dev so bear with me)

`pkg . --target node18-linux-x64 --output dejpeg-node`