/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE run_graph_test

#include "../core/common.hpp"

#include <cxtream/tensorflow/load_graph.hpp>
#include <cxtream/tensorflow/run_graph.hpp>

#include <boost/test/unit_test.hpp>

using namespace cxtream::tensorflow;
using namespace boost;

BOOST_AUTO_TEST_CASE(test_simple_run)
{
    auto sess = load_graph("transpose_add_one_2_3_net.pb");
    // prepare inputs
    std::vector<std::string>       input_names  = {"input"};
    std::tuple<std::vector<float>> input_data   = {{0, 1, 2, 3, 4, 5}};
    std::vector<std::vector<long>> input_shapes = {{2, 3}};
    std::vector<std::string>       output_names = {"output"};

    // prepare outputs
    std::tuple<std::vector<float>> output_data;
    std::vector<std::vector<long>> output_shapes;

    // run graph
    std::tie(output_data, output_shapes) =
      run_graph<float>(*sess, input_names, input_data, input_shapes, output_names);

    // test output shape
    BOOST_TEST(output_shapes.size() == 1);
    BOOST_TEST(output_shapes[0][0] == 3);
    BOOST_TEST(output_shapes[0][1] == 2);

    // test output data
    test_ranges_equal(std::get<0>(output_data), std::vector<float>{1, 4, 2, 5, 3, 6});
}
