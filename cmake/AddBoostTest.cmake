function(add_boost_test EXECUTABLE_FILE_NAME SOURCE_FILE_NAME LIBRARIES)
  add_executable(
    ${EXECUTABLE_FILE_NAME}
    ${SOURCE_FILE_NAME}
  )

  target_link_libraries(
    ${EXECUTABLE_FILE_NAME} 
    cxtream_core
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    ${LIBRARIES}
  )

  target_include_directories(
    ${EXECUTABLE_FILE_NAME}
    PRIVATE ${Boost_INCLUDE_DIRS}
  )

  # find all BOOST_AUTO_TEST_CASE(*) lines
  file(READ "${SOURCE_FILE_NAME}" _source_file_content)
  string(
    REGEX MATCHALL "BOOST_AUTO_TEST_CASE\\( *([A-Za-z_0-9]+) *\\)" 
    _boost_tests ${_source_file_content}
  )

  foreach(_test_raw_name ${_boost_tests})
    # take the name of the test
    string(
      REGEX REPLACE ".*\\( *([A-Za-z_0-9]+) *\\).*" "\\1"
      _test_name ${_test_raw_name}
    )
    # register the test
    add_test(
      NAME "${EXECUTABLE_FILE_NAME}.${_test_name}"
      COMMAND ${EXECUTABLE_FILE_NAME} --run_test=${_test_name}
      --catch_system_error=yes
    )
  endforeach()
endfunction()
