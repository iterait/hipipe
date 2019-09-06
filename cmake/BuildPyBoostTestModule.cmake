function(build_pyboost_test_module TARGET_PREFIX MODULE_NAME SOURCE_FILE)

  add_library(
    "${TARGET_PREFIX}.${MODULE_NAME}" SHARED
    "${SOURCE_FILE}"
  )

  set_target_properties(
    "${TARGET_PREFIX}.${MODULE_NAME}" PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${MODULE_NAME}"
    DEBUG_POSTFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )

  target_link_libraries(
    "${TARGET_PREFIX}.${MODULE_NAME}"
    PRIVATE hipipe_core
  )

endfunction()
