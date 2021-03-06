/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE load_graph_test

#include <hipipe/tensorflow/load_graph.hpp>

#include <boost/test/unit_test.hpp>

namespace htf = hipipe::tensorflow;

BOOST_AUTO_TEST_CASE(test_simple_load)
{
    auto sess = htf::load_graph("transpose_add_one_2_3_net.pb");
    BOOST_CHECK(sess);
    // further testing performed in run_graph.cpp
}
