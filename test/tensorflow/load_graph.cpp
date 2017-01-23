/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE load_graph_test

#include <cxtream/tensorflow/load_graph.hpp>

#include <boost/test/unit_test.hpp>

using namespace cxtream::tensorflow;
using namespace boost;

BOOST_AUTO_TEST_CASE(test_simple_load)
{
    auto sess = load_graph("transpose_add_one_2_3_net.pb");
    BOOST_CHECK(sess);
    // further testing performed in run_graph.cpp
}
