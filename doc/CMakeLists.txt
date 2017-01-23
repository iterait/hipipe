cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

configure_file(Doxyfile.in Doxyfile @ONLY)

add_custom_target(
  doc
  COMMAND "${CMAKE_COMMAND}" -E remove_directory html
  COMMAND "${DOXYGEN_EXECUTABLE}" Doxyfile
  VERBATIM
)

execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
  OUTPUT_VARIABLE CXTREAM_GIT_HEAD_SHORT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_custom_target(doc.deploy
  # checkout gh-pages branch
  COMMAND "${GIT_EXECUTABLE}" checkout gh-pages
  # pull
  COMMAND "${GIT_EXECUTABLE}" pull
  # clean
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_LIST_DIR}/clean-gh-pages.cmake"
  # copy
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/html"
                                               "${CMAKE_CURRENT_LIST_DIR}/gh-pages"
  # commit
  COMMAND "${GIT_EXECUTABLE}" add --all
  COMMAND "${GIT_EXECUTABLE}" commit -m "Update docs (commit ${CXTREAM_GIT_HEAD_SHORT_SHA})"
  # push
  COMMAND "${GIT_EXECUTABLE}" push
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/gh-pages"
  DEPENDS doc
  VERBATIM
)
