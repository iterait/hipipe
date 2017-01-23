# Remove files mentioned in 'install_manifest.txt'

set(_manifest "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(EXISTS ${_manifest})
  message(STATUS "Uninstalling...")

  file(STRINGS ${_manifest} _files)

  foreach(_file ${_files})
    if(IS_SYMLINK "${_file}" OR EXISTS "${_file}")
      message(STATUS "Removing ${_file}")

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove "${_file}"
        OUTPUT_VARIABLE _rm_output
        RESULT_VARIABLE _rm_retval
      )

      if(NOT "${_rm_retval}" STREQUAL 0)
        message(FATAL_ERROR "Unable to remove ${_file}")
      endif()

    else()
      message(STATUS "File ${_file} does not exist")
    endif()
  endforeach()

  message(STATUS "Uninstallation successful")

else()
  message(STATUS "Manifest file '${_manifest}' does not exist.")
  message(STATUS "The project is probably not installed.")
endif()
