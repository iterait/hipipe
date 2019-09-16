cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(
  "hipipe"
  VERSION 0.7.1
)
set(PROJECT_DESCRIPTION "C++17 data pipeline with Python bindings.")

# -------
# Options
# -------

option(HIPIPE_BUILD_TEST "Build test binaries" ON)
option(HIPIPE_BUILD_DOC "Build documentation" OFF)
option(HIPIPE_BUILD_PYTHON "Build C++ <-> Python converters" ON)
option(HIPIPE_BUILD_PYTHON_OPENCV "Build C++ <-> Python OpenCV converters (requires HIPIPE_BUILD_PYTHON)" ON)
option(HIPIPE_BUILD_TENSORFLOW "Build TensorFlow functionality" OFF)
option(HIPIPE_BUILTIN_RANGEV3 "Use built-in Range-v3 library" ON)

# -------------
# CMake Options
# -------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
# add 'd' suffix to libraries in debug mode
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

# -------------------------------------
# Dump CMake Options Into a Header File
# -------------------------------------

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/hipipe/build_config.hpp"
)

# --------------
# Compiler Flags
# --------------

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# ----------------------
# Shorter Error Messages
# ----------------------

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "-Wfatal-errors ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # GCC truncates multiline errors with -Wfatal-errors
  # using -fmax-erorrs instead
  set(CMAKE_CXX_FLAGS "-fmax-errors=2 ${CMAKE_CXX_FLAGS}")
endif()

# ---------------------
# Find Common Libraries
# ---------------------

if(HIPIPE_BUILD_PYTHON)
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
  find_package(PythonNumPy REQUIRED)
  find_package(BoostPython3 REQUIRED)

  if(HIPIPE_BUILD_PYTHON_OPENCV)
    find_package(OpenCV COMPONENTS core REQUIRED)
  endif()
endif()

if(HIPIPE_BUILD_DOC)
  find_package(Git REQUIRED)
  find_package(Doxygen REQUIRED)
endif()

if(HIPIPE_BUILD_TEST)
  find_package(Boost 1.61 COMPONENTS system thread unit_test_framework REQUIRED)
else()
  find_package(Boost 1.61 COMPONENTS system thread REQUIRED)
endif()

# ------------------
# Build hipipe core
# ------------------

file(GLOB_RECURSE
  hipipe_core_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
)
add_library(
  hipipe_core SHARED
  ${hipipe_core_sources}
)
set_target_properties(
  hipipe_core PROPERTIES
    EXPORT_NAME Core
    INTERFACE_COMPILE_OPTIONS "-std=c++17"
    VERSION ${PROJECT_VERSION}
)
target_link_libraries(
  hipipe_core
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC pthread
  PUBLIC stdc++fs
)
target_include_directories(
  hipipe_core
  PUBLIC ${Boost_INCLUDE_DIRS}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # build_config.hpp is here
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)

# -----------------------------
# Use Range-v3 from a submodule
# -----------------------------

if(HIPIPE_BUILTIN_RANGEV3)
  target_include_directories(
    hipipe_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/range-v3/include>
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/hipipe/third_party/range-v3>
  )
endif()

# --------------------------------------
# Build hipipe::core with Python support
# --------------------------------------

if(HIPIPE_BUILD_PYTHON)
  target_include_directories(
    hipipe_core
    PUBLIC ${PYTHON_INCLUDE_DIRS}
    PUBLIC ${PYTHON_NUMPY_INCLUDE_DIR}
    PUBLIC ${BoostPython3_INCLUDE_DIRS}
  )
  target_link_libraries(
    hipipe_core
    PUBLIC ${PYTHON_LIBRARIES}
    PUBLIC ${BoostPython3_LIBRARIES}
  )

  # -------------------------------------------------
  # Build hipipe::core with Python and OpenCV support
  # -------------------------------------------------

  if(HIPIPE_BUILD_PYTHON_OPENCV)
    target_include_directories(
      hipipe_core
      PUBLIC ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(
      hipipe_core
      PUBLIC ${OpenCV_LIBRARIES}
    )
  endif()
endif()


# -------------------------
# Build hipipe::tensorflow
# -------------------------

if(HIPIPE_BUILD_TENSORFLOW)
  file(GLOB_RECURSE
    hipipe_tensorflow_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensorflow/*.cpp"
  )
  add_library(
    hipipe_tensorflow SHARED
    ${hipipe_tensorflow_sources}
  )
  set_target_properties(
    hipipe_tensorflow PROPERTIES EXPORT_NAME Tensorflow
  )
  target_link_libraries(
    hipipe_tensorflow
    PUBLIC hipipe_core
  )
  # try the shared tensorflow library first
  find_package(TensorflowCC COMPONENTS Shared)
  if(TensorflowCC_Shared_FOUND)
    target_link_libraries(
      hipipe_tensorflow
      PUBLIC TensorflowCC::Shared
    )
    set_target_properties(
      TensorflowCC::Shared
      PROPERTIES INTERFACE_COMPILE_OPTIONS "-std=c++17"
    )
  # fallback to the static library
  # static library is linked privately to avoid multiple funtion definitions,
  # but the include directories are propagated as public
  else()
    find_package(TensorflowCC REQUIRED COMPONENTS Static)
    target_include_directories(
      hipipe_tensorflow
      PUBLIC "$<TARGET_PROPERTY:TensorflowCC::Static,INTERFACE_INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(
      hipipe_tensorflow
      PRIVATE TensorflowCC::Static
    )
    set_target_properties(
      TensorflowCC::Static
      PROPERTIES INTERFACE_COMPILE_OPTIONS "-std=c++17"
    )
  endif()
  set_target_properties(
    hipipe_tensorflow
    PROPERTIES VERSION ${PROJECT_VERSION}
  )
endif()

# -----
# Tests
# -----

if(HIPIPE_BUILD_TEST)
  enable_testing()
  include("AddBoostTest")
  if(HIPIPE_BUILD_PYTHON)
      include("AddPythonTest")
      include("BuildPyBoostTestModule")
  endif()
  add_subdirectory("test")
endif()

# -------------
# Documentation
# -------------

if(HIPIPE_BUILD_DOC)
  add_subdirectory(doc)
endif()

# -------
# Install
# -------

# install built-in range-v3 library
if(HIPIPE_BUILTIN_RANGEV3)
  install(
    DIRECTORY third_party/range-v3/include/
    DESTINATION include/hipipe/third_party/range-v3
  )
endif()

# install core header files
install(
  DIRECTORY include/hipipe/core/
  DESTINATION include/hipipe/core
)
install(
  FILES include/hipipe/core.hpp
  DESTINATION include/hipipe
)
install(
  TARGETS hipipe_core
  EXPORT HiPipeCoreTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
# install generated files (e.g., build_config.hpp)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION include
)

# install hipipe::tensorflow library
if(HIPIPE_BUILD_TENSORFLOW)
  install(
    DIRECTORY include/hipipe/tensorflow/
    DESTINATION include/hipipe/tensorflow
  )
  install(
    FILES include/hipipe/tensorflow.hpp
    DESTINATION include/hipipe
  )
  install(
    TARGETS hipipe_tensorflow
    EXPORT HiPipeTensorflowTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

# ---------
# Uninstall
# ---------

add_custom_target(
  uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake
)

# ------------------------------
# Cmake Config and Version Files
# ------------------------------

include(CMakePackageConfigHelpers)
set(CMAKECFG_INSTALL_DIR lib/cmake/HiPipe)

# configure package files
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HiPipeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HiPipeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HiPipeConfig.cmake"
  INSTALL_DESTINATION "${CMAKECFG_INSTALL_DIR}"
  NO_SET_AND_CHECK_MACRO # HiPipe only uses interface libraries
)

# install the core targets file
install(
  EXPORT HiPipeCoreTargets
  FILE HiPipeCoreTargets.cmake
  NAMESPACE HiPipe::
  DESTINATION "${CMAKECFG_INSTALL_DIR}"
)
# export the targets also for local use
export(
  TARGETS hipipe_core
  FILE HiPipeCoreTargets.cmake
  NAMESPACE HiPipe::
)

# install the tensorflow targets file
if(HIPIPE_BUILD_TENSORFLOW)
  install(
    EXPORT HiPipeTensorflowTargets
    FILE HiPipeTensorflowTargets.cmake
    NAMESPACE HiPipe::
    DESTINATION "${CMAKECFG_INSTALL_DIR}"
  )
  # export the targets also for local use
  export(
    TARGETS hipipe_tensorflow
    FILE HiPipeTensorflowTargets.cmake
    NAMESPACE HiPipe::
  )
endif()

# install the package files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HiPipeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HiPipeConfigVersion.cmake
  DESTINATION ${CMAKECFG_INSTALL_DIR}
)
