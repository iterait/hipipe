# we need to know the python version
if(NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp 3 REQUIRED)
endif()

# find the matching boost python implementation
set(_python_version ${PYTHONLIBS_VERSION_STRING})

# remove the minor-most version number until the boost component is found
while(NOT BoostPython3_FOUND)

  # boost does not separate version numbers with dots
  STRING(REGEX REPLACE "[^0-9]" "" _boost_python_version ${_python_version})

  # find component python-pyXX (Ubuntu, Debian...)
  message(STATUS "Searching for Boost component python-py${_boost_python_version}")
  find_package(Boost COMPONENTS "python-py${_boost_python_version}")
  set(BoostPython3_FOUND ${Boost_PYTHON-PY${_boost_python_version}_FOUND})

  # find component pythonXX (Arch...)
  message(STATUS "Searching for Boost component python${_boost_python_version}")
  if(NOT BoostPython3_FOUND)
    find_package(Boost COMPONENTS "python${_boost_python_version}")
    set(BoostPython3_FOUND ${Boost_PYTHON${_boost_python_version}_FOUND})
  endif()

  # break if there are no minor versions to cut off
  STRING(REGEX MATCH "([0-9.]+).[0-9]+" _has_minor ${_python_version})
  if("${_has_minor}" STREQUAL "")
    break()
  else()
    STRING(REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" _python_version ${_python_version})
  endif()

endwhile()

if(BoostPython3_FOUND)
  set(BoostPython3_LIBRARIES ${Boost_LIBRARIES})
  set(BoostPython3_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  BoostPython3 DEFAULT_MSG
  BoostPython3_FOUND
)
