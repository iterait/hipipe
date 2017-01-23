configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/transpose_add_one_2_3_net/transpose_add_one_2_3_net.pb"
  "${CMAKE_CURRENT_BINARY_DIR}/transpose_add_one_2_3_net.pb"
  COPYONLY
)

add_boost_test("test.tensorflow.load_graph" "load_graph.cpp" "cxtream_tensorflow")

add_boost_test("test.tensorflow.run_graph" "run_graph.cpp" "cxtream_tensorflow")
