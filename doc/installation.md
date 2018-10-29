Installation {#installation}
============

Requirements
------------
---

Officially supported systems are Ubuntu 18.04+ and Arch Linux, although __hipipe__ should
work on any recent enough system. The __hipipe core__ is a pure C++ library that by default 
depends on [Boost C++ Libraries](http://www.boost.org/) (v1.61+ with Boost::Python
is required) and [OpenCV](http://opencv.org/) for image conversion between C++ and Python.
Python bindings and OpenCV support can be disabled, see Advanced Build Options section below.

If you plan to use the full functionality (this is the default behavior),
install all the requirements by one of the following commands:

```
# Arch Linux
pacman -S git base-devel cmake boost opencv python python-numpy

# Ubuntu 18.04+
apt install git build-essential cmake libboost-all-dev libopencv-dev python3-dev python3-numpy
```

If you plan to use [TensorFlow C++ API](https://www.tensorflow.org/api_guides/cc/guide),
please install the TensorFlow C++ library into your system using
[tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc) project. This is not neccessary
if you only want to use TensorFlow in Python and it is disabled by default.

Download
--------
---

The complete source code can be downloaded from our official GitHub
[repository](https://github.com/iterait/hipipe) using the following commands:

```
git clone --recursive https://github.com/iterait/hipipe.git
cd hipipe
```

Build & Install
---------------
---

Use the following for system-wide installation:

```
mkdir build && cd build
cmake ..
make -j4
ctest --output-on-failure
sudo make install
```

Or use the following for userspace installation:

```
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=~/.local ..
make -j4
ctest --output-on-failure
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

| Option                     | Description                                                                   | Default      |
|----------------------------|-------------------------------------------------------------------------------|--------------|
| HIPIPE_BUILD_TEST          | Build tests.                                                                  | ON           |
| HIPIPE_BUILD_DOC           | Build documentation.                                                          | OFF          |
| HIPIPE_BUILD_PYTHON        | Build Python functionality.                                                   | ON           |
| HIPIPE_BUILD_PYTHON_OPENCV | Build Python OpenCV converters (requires HIPIPE_BUILD_PYTHON).                | ON           |
| HIPIPE_BUILD_TENSORFLOW    | Build TensorFlow functionality (unnecessary if you use TensorFlow in Python). | OFF          |
| HIPIPE_BUILTIN_RANGEV3     | Install and use the built-in Range-v3 library.                                | ON           |
| CMAKE_INSTALL_PREFIX       | The path where hipipe will be installed.                                      | OS-dependent |
| CMAKE_CXX_COMPILER         | The compiler command to be used, e.g., g++ or clang++.                        | OS-dependent |

Development
-----------
---

Are you missing a feature or have you found a bug? You can either fill an issue on our
GitHub (https://github.com/iterait/hipipe) or even better, become a developer!

There are no special requirements for development, just a few hints:
- Header files are under `include/` directory.
- Source files are under `src/` directory.
- Unit tests are under `test/` directory.
- Any new functionality has to be accompanied by unit tests.
- If you add a new header, don't forget to add it to the corresponding accumulating header in parent folder.
  For instance, `include/core/stream.hpp` lists all the headers under `include/core/stream` directory.
- If you add a new test, don't forget to register it in `CMakeLists.txt` located in the same folder as the test.
- To build and run a single unit test, run e.g.,:
```
make test.core.stream.drop
ctest --output-on-failure -R test.core.stream.drop
```
- To build and run a single test that requires building a Python module, run e.g.,:
```
make test.core.python.stream.converter_py_cpp  # note the py_cpp suffix
ctest --output-on-failure -R test.core.python.stream.converter
```
