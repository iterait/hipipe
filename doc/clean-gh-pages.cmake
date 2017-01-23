# remove all files except for the hidden ones
file(GLOB _gh_files RELATIVE "${CMAKE_CURRENT_LIST_DIR}/gh-pages" "[^.]*")
if(_gh_files)
  # do not remove GitHub's CNAME file
  list(REMOVE_ITEM _gh_files "CNAME")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E remove ${_gh_files})
  foreach(_file ${_gh_files})
    execute_process(COMMAND "${CMAKE_COMMAND}" -E remove_directory ${_file})
  endforeach()
endif()
