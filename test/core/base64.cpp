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
#define BOOST_TEST_MODULE base64_test

#include <hipipe/core/base64.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/all.hpp>

#include <vector>

namespace hp = hipipe;

std::vector<std::vector<unsigned char>> data{
  {},
  {0},
  {0, 120},
  {0, 120, 15},
  {0, 120, 15, 10},
  {0, 120, 15, 10, 0},
  {0, 120, 15, 10, 0, 0},
  {0, 120, 15, 10, 0, 0, 0},
  {0, 120, 15, 10, 0, 0, 0, 0},
};

std::vector<std::string> base64_data{
 "",
 "AA==",
 "AHg=",
 "AHgP",
 "AHgPCg==",
 "AHgPCgA=",
 "AHgPCgAA",
 "AHgPCgAAAA==",
 "AHgPCgAAAAA="
};

BOOST_AUTO_TEST_CASE(test_b64_encode)
{
    for (std::size_t i = 0; i < data.size(); ++i) {
        BOOST_TEST(hp::base64_encode(data[i]) == base64_data[i]);
    }
}

BOOST_AUTO_TEST_CASE(test_b64_decode)
{
    for (std::size_t i = 0; i < data.size(); ++i) {
        BOOST_TEST(hp::base64_decode(base64_data[i]) == data[i]);
    }
}
