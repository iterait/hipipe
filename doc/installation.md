Installation {#installation}
============

Requirements
------------
---

Officially supported systems are Ubuntu 16.10+ and Arch Linux, although __cxtream__ should
work on any recent enough system. The __cxtream core__ is a pure C++ library with a
single dependency to [Boost C++ Libraries](http://www.boost.org/)
(Boost 1.61+ is required). Extensions to the core library are __Python bindings__ with
automatic [OpenCV](http://opencv.org/) image conversion between C++ and Python.

If you plan to use the full functionality (this is the default behavior),
install all the requirements by one of the following commands:

```
# Arch Linux
pacman -S git base-devel cmake boost opencv python python-numpy

# Ubuntu 16.10+
apt install git build-essential cmake libboost-all-dev libopencv-dev python3-dev python3-numpy
```

If you want to use __cxtream core__ only, use one of the following instead:

```
# Arch Linux
pacman -S git base-devel cmake boost

# Ubuntu 16.10+
apt install git build-essential cmake libboost-all-dev
```

If you plan to use [TensorFlow C++ API](https://www.tensorflow.org/api_guides/cc/guide),
please install the TensorFlow C++ library into your system using
[tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc) project. This is not neccessary
if you only want to use TensorFlow in Python and it is disabled by default.

Download
--------
---

The complete source code can be downloaded from our official GitHub
[repository](https://github.com/Cognexa/cxtream) using the following commands:

```
git clone --recursive https://github.com/Cognexa/cxtream.git
cd cxtream
```

Build & Install
---------------
---

Use the following for system-wide installation:

```
mkdir build && cd build
cmake ..
make -j5
make test
sudo make install
```

Or use the following for userspace installation:

```
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=~/.local ..
make -j5
make test
make install
```

For userspace installation, don't forget to set the appropriate
environmental variables, e.g., add the following to your `.bashrc` / `.zshrc`:
```
# register ~/.local system hierarchy
export PATH="${HOME}/.local/bin:$PATH"
export LIBRARY_PATH="${HOME}/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="${HOME}/.local/lib:$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="${HOME}/.local/include:$CPLUS_INCLUDE_PATH"
```

Advanced Build Options
----------------------
---

There are multiple build options that can be set when configuring the project
with CMake. For instance, if you don't want to build tests because it takes a
lot of time, and you are also not interested in Python interoperability,
you may use the `BUILD_TEST` and `BUILD_PYTHON` flags as follows:

```
cmake -DBUILD_TEST=OFF -DBUILD_PYTHON=OFF ..
```

The full list of supported options is the following:

| Option               | Description                                                                   | Default      |
|----------------------|-------------------------------------------------------------------------------|--------------|
| BUILD_TEST           | Build tests.                                                                  | ON           |
| BUILD_DOC            | Build documentation.                                                          | OFF          |
| BUILD_PYTHON         | Build Python functionality.                                                   | ON           |
| BUILD_PYTHON_OPENCV  | Build Python OpenCV converters (requires BUILD_PYTHON).                       | ON           |
| BUILD_TENSORFLOW     | Build TensorFlow functionality (unnecessary if you use TensorFlow in Python). | OFF          |
| BUILTIN_RANGEV3      | Install and use the built-in Range-v3 library.                                | ON           |
| CMAKE_INSTALL_PREFIX | The path where cxtream will be installed.                                     | OS-dependent |
| CMAKE_CXX_COMPILER   | The compiler command to be used, e.g., g++ or clang++.                        | OS-dependent |
