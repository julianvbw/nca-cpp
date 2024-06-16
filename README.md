# Growing Neural Cellular Automata in C++
A reproduction of Growing Neural Cellular Automata (NCA) in C++17 using LibTorch and GLFW/OpenGL. The whole project is based on [this great distill pub article from 2020](https://distill.pub/2020/growing-ca/).

My main aim was to try out the LibTorch API out of curiosity, while producing something nice-looking and interactive. It also features a minimal UI that uses CUDA OpenGL interoperation to move tensors from LibTorch to an OpenGL texture, which is really neat imo. I hope this project can provide help to someone who is struggling with LibTorch or the concept of NCAs. Or just someone who is easily entertained by watching cellular automata.

This project comes in two parts: A training executable  with an extensive CLI build on [TCLAP](https://tclap.sourceforge.net/) and an interactive GUI based on [GLFW](https://github.com/glfw/glfw). Essential are of course [LibTorch](https://pytorch.org/cppdocs/) and [LodePNG](https://github.com/lvandeve/lodepng) for PNG file handling in C++.

This project comes with an example image, the 32x32px dragon emoji from [Google's Noto Emoji repository](https://github.com/googlefonts/noto-emoji) (Happy Year of the Dragon), and a fully trained model using the example parameters below.
[](example/grow.gif)

# How to build
First of all, this repository comes with all dependencies *except* LibTorch, since it is huge. Go get it yourself and make sure the __CMAKE_PREFIX_PATH__ is set correctly!

Then you may build using:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=[ABSOLUTE_PATH_TO_LIBTORCH]
cmake --build . --config Release [--target nca_train OR nca_gui DEFAULT all] [-- -j NUMBER_OF_THREADS]
```

This will build the binaries `nca_train` and, if OpenGL is available, `nca_gui`. If you're building `nca_gui`, also have CUDA ready, since this project uses OpenGL interop to avoid device->host copies in between updates.

# How to use
All arguments with a short description are found by starting the binary with `--help`.

Example usage:
```
./nca_train -i ../example/dragon.png -o ../example/output/ --epochs 8000 -R 0.001 -S 2000 -c 32 -m 96 -M 128 -v -v -v
```
will create an output folder with the model, an image of the best training state every 20 training steps and an evaluation growth run of 400 updates/images that can be converted to a gif. This example can take long and is a pretty GPU-intensive choice of epochs.

Note: the trailing comma for the output folder is important! Otherwise `output` would be considered as the model output filename and use the current path as the folder.

```
./nca_gui ../example/output/model-dragon.lth -H 128 -W 128
```
will run the trained NCA on a 128x128px canvas. Drag your mouse while left-clicking to damage the state. Press R on the keyboard to restart/reseed.

# Compatibility
This _should_ be cross-platform compatible, so you can try on Windows or MacOS if you're brave. I personally have not.

# Why?
It's fun.
