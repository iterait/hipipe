cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(
  "cxtream"
  VERSION 0.5.1
)
set(PROJECT_DESCRIPTION "C++17 data pipeline with Python bindings.")

# -------
# Options
# -------

option(BUILD_TEST "Build test binaries" ON)
option(BUILD_DOC "Build documentation" OFF)
option(BUILD_PYTHON "Build C++ <-> Python converters" ON)
option(BUILD_PYTHON_OPENCV "Build C++ <-> Python OpenCV converters (requires BUILD_PYTHON)" ON)
option(BUILD_TENSORFLOW "Build TensorFlow functionality" OFF)
option(BUILTIN_RANGEV3 "Use built-in Range-v3 library" ON)

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
  "${CMAKE_CURRENT_BINARY_DIR}/include/cxtream/build_config.hpp"
)

# --------------
# Compiler Flags
# --------------

set(CMAKE_CXX_FLAGS "-std=c++1z -Wall -pedantic ${CMAKE_CXX_FLAGS}")
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

if(BUILD_PYTHON)
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
  find_package(BoostPython3 REQUIRED)

  if(BUILD_PYTHON_OPENCV)
    find_package(PythonNumPy REQUIRED)
    find_package(OpenCV COMPONENTS core REQUIRED)
  endif()
endif()

if(BUILD_DOC)
  find_package(Git REQUIRED)
  find_package(Doxygen REQUIRED)
endif()

if(BUILD_TEST)
  find_package(Boost 1.61 COMPONENTS system thread unit_test_framework REQUIRED)
else()
  find_package(Boost 1.61 COMPONENTS system thread REQUIRED)
endif()

# ------------------
# Build cxtream core
# ------------------

add_library(cxtream_core INTERFACE)
set_target_properties(
  cxtream_core PROPERTIES EXPORT_NAME Core
)
target_compile_options(
  cxtream_core INTERFACE
  "-std=c++1z"
)
target_link_libraries(
  cxtream_core INTERFACE
  ${Boost_LIBRARIES}
  pthread
  stdc++fs
)
target_include_directories(
  cxtream_core INTERFACE
  ${Boost_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # build_config.hpp is here
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
if(BUILTIN_RANGEV3)
  target_include_directories(
    cxtream_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/range-v3/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/cxtream/third_party/range-v3>
  )
endif()

# ---------------------
# Build cxtream::python
# ---------------------

if(BUILD_PYTHON)

  # -------------------------
  # Build cxtream::python .so
  # -------------------------

  add_library(
    cxtream_python SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/python/initialize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/python/utility/pyboost_fs_path_converter.cpp"
  )
  set_target_properties(
    cxtream_python PROPERTIES EXPORT_NAME Python
  )
  target_include_directories(
    cxtream_python
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${PYTHON_INCLUDE_DIRS}
    PUBLIC ${BoostPython3_INCLUDE_DIRS}
  )
  target_link_libraries(
    cxtream_python
    cxtream_core
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${BoostPython3_LIBRARIES}
    stdc++fs
  )
  set_target_properties(
    cxtream_python
    PROPERTIES VERSION ${PROJECT_VERSION}
  )

  # -----------------------------------------------------
  # Build cxtream::python::utility::pyboost_cv3_converter
  # -----------------------------------------------------

  if(BUILD_PYTHON_OPENCV)
    target_sources(
      cxtream_python
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/python/utility/pyboost_cv2_converter.cpp"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/python/utility/pyboost_cv3_converter.cpp"
    )
    target_include_directories(
      cxtream_python
      PUBLIC ${PYTHON_NUMPY_INCLUDE_DIRS}
      PUBLIC ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(
      cxtream_python
      ${OpenCV_LIBRARIES}
    )
  endif()

endif()

# -------------------------
# Build cxtream::tensorflow
# -------------------------

if(BUILD_TENSORFLOW)
  add_library(
    cxtream_tensorflow SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensorflow/load_graph.cpp"
  )
  set_target_properties(
    cxtream_tensorflow PROPERTIES EXPORT_NAME Tensorflow
  )
  target_link_libraries(
    cxtream_tensorflow
    PUBLIC cxtream_core
    PUBLIC stdc++fs
  )
  # try the shared tensorflow library first
  find_package(TensorflowCC COMPONENTS Shared)
  if(TensorflowCC_Shared_FOUND)
    target_link_libraries(
      cxtream_tensorflow
      PUBLIC TensorflowCC::Shared
    )
    set_target_properties(
      TensorflowCC::Shared
      PROPERTIES INTERFACE_COMPILE_OPTIONS "-std=c++1z"
    )
  # fallback to the static library
  # static library is linked privately to avoid multiple funtion definitions,
  # but the include directories are propagated as public
  else()
    find_package(TensorflowCC REQUIRED COMPONENTS Static)
    target_include_directories(
      cxtream_tensorflow
      PUBLIC "$<TARGET_PROPERTY:TensorflowCC::Static,INTERFACE_INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(
      cxtream_tensorflow
      PRIVATE TensorflowCC::Static
    )
    set_target_properties(
      TensorflowCC::Static
      PROPERTIES INTERFACE_COMPILE_OPTIONS "-std=c++1z"
    )
  endif()
  set_target_properties(
    cxtream_tensorflow
    PROPERTIES VERSION ${PROJECT_VERSION}
  )
endif()

# -----
# Tests
# -----

if(BUILD_TEST)
  enable_testing()
  include("AddBoostTest")
  if(BUILD_PYTHON)
      include("AddPythonTest")
      include("BuildPyBoostTestModule")
  endif()
  add_subdirectory("test")
endif()

# -------------
# Documentation
# -------------

if(BUILD_DOC)
  add_subdirectory(doc)
endif()

# -------
# Install
# -------

# install built-in range-v3 library
if(BUILTIN_RANGEV3)
  install(
    DIRECTORY third_party/range-v3/include/
    DESTINATION include/cxtream/third_party/range-v3
  )
endif()

# install core header files
install(
  DIRECTORY include/cxtream/core/
  DESTINATION include/cxtream/core
)
install(
  FILES include/cxtream/core.hpp
  DESTINATION include/cxtream
)

# install generated files (e.g., build_config.hpp)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION include
)

# install cxtream::core target
install(
  TARGETS cxtream_core
  EXPORT CxtreamCoreTargets
)

# install cxtream::python library
if(BUILD_PYTHON)
  install(
    DIRECTORY include/cxtream/python/
    DESTINATION include/cxtream/python
  )
  install(
    FILES include/cxtream/python.hpp
    DESTINATION include/cxtream
  )
  install(
    TARGETS cxtream_python
    EXPORT CxtreamPythonTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

# install cxtream::tensorflow library
if(BUILD_TENSORFLOW)
  install(
    DIRECTORY include/cxtream/tensorflow/
    DESTINATION include/cxtream/tensorflow
  )
  install(
    FILES include/cxtream/tensorflow.hpp
    DESTINATION include/cxtream
  )
  install(
    TARGETS cxtream_tensorflow
    EXPORT CxtreamTensorflowTargets
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
set(CMAKECFG_INSTALL_DIR lib/cmake/Cxtream)

# configure package files
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CxtreamConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CxtreamConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CxtreamConfig.cmake"
  INSTALL_DESTINATION "${CMAKECFG_INSTALL_DIR}"
  NO_SET_AND_CHECK_MACRO # Cxtream only uses interface libraries
)

# install the core targets file
install(
  EXPORT CxtreamCoreTargets
  FILE CxtreamCoreTargets.cmake
  NAMESPACE Cxtream::
  DESTINATION "${CMAKECFG_INSTALL_DIR}"
)
# export the targets also for local use
export(
  TARGETS cxtream_core
  FILE CxtreamCoreTargets.cmake
  NAMESPACE Cxtream::
)

# install the python targets file
if(BUILD_PYTHON)
  install(
    EXPORT CxtreamPythonTargets
    FILE CxtreamPythonTargets.cmake
    NAMESPACE Cxtream::
    DESTINATION "${CMAKECFG_INSTALL_DIR}"
  )
  # export the targets also for local use
  export(
    TARGETS cxtream_python
    FILE CxtreamPythonTargets.cmake
    NAMESPACE Cxtream::
  )
endif()

# install the tensorflow targets file
if(BUILD_TENSORFLOW)
  install(
    EXPORT CxtreamTensorflowTargets
    FILE CxtreamTensorflowTargets.cmake
    NAMESPACE Cxtream::
    DESTINATION "${CMAKECFG_INSTALL_DIR}"
  )
  # export the targets also for local use
  export(
    TARGETS cxtream_tensorflow
    FILE CxtreamTensorflowTargets.cmake
    NAMESPACE Cxtream::
  )
endif()

# install the package files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CxtreamConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CxtreamConfigVersion.cmake
  DESTINATION ${CMAKECFG_INSTALL_DIR}
)
