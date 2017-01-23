cmake_minimum_required(VERSION 3.0)

#############################
# Set the following variables
# PYTHON_NUMPY_INCLUDE_DIR
# PYTHON_NUMPY_FOUND
#############################

if(NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp REQUIRED)
endif()

# Find out the include path
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
  OUTPUT_VARIABLE _python_numpy_path
)
# And the version
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
  OUTPUT_VARIABLE _python_numpy_version
)

find_path(
  PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${_python_numpy_path}" "${PYTHON_INCLUDE_PATH}"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  PYTHON_NUMPY
  REQUIRED_VARS PYTHON_NUMPY_INCLUDE_DIR
  VERSION_VAR _numpy_version
)
