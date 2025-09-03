### low development stage (releases will be slow)

minimal desktop version of DeJPEG supporting *only* for FBCNN and SCUNet

use [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) for more complex processing and GPU support.

prebuilt binaries available under `Releases` or build below

# usage
prerequisites:

nodejs 18~20

```bash
npm install
npm run build && npm start
```

# building

## Windows

note: the dev system is Linux, so `npm run build` wont work out of the box, and you will need to copy `src/static` to `dist`

requires [microsoft vc C++ redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version)

`pkg . --target node18-win-x64 --output dejpeg-windows.exe`

## Linux

(tested on Arch)

rename `node_modules` to `node_modules_linux` (and if you have a windows version of `node_modules`, it should be `node_modules_windows`) then:

```bash
npm run build-<platform>
```

where <platform> is either `linux` or `windows`

this will build and create a compressed 7z (if 7zip installed)

more info below

it is possible to cross-platform build, and since i dont use windows i used a light windows vm, installed nodejs ran `npm install` and copied `node_modules` to my linux environment and rename/replace linux's `node_modules` folder with the windows version, then run a windows build command. so now I dont need to use a windows vm when releasing a new version yay, this is what `scripts/build-(platform).sh` is for. I don't use Windows therefore no batch/powershell versions for these

### disclaimer

im not a full-stack developer nor claim to be one, this is a hobby project and the code may be messy and the setup may not follow best practices.
