add_subdirectory("core")

if(BUILD_TENSORFLOW)
  add_subdirectory("tensorflow")
endif()

if(BUILD_PYTHON)
  add_subdirectory("python")
endif()
