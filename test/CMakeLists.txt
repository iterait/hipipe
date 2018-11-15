add_subdirectory("core")

if(HIPIPE_BUILD_TENSORFLOW)
  add_subdirectory("tensorflow")
endif()
