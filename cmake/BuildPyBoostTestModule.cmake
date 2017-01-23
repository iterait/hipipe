function(build_pyboost_test_module MODULE_NAME SOURCE_FILE)

  add_library(
    "${MODULE_NAME}" SHARED
    "${SOURCE_FILE}"
  )

  set_target_properties(
    "${MODULE_NAME}" PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  
  target_link_libraries(
    "${MODULE_NAME}"
    PRIVATE cxtream_python
  )

endfunction()
