function(add_python_test TEST_NAME SCRIPT_NAME)

  configure_file(
    "${SCRIPT_NAME}"
    "${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_NAME}"
    COPYONLY
  )

  add_test(
    NAME "${TEST_NAME}"
    COMMAND "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT_NAME}"
  )

endfunction()
