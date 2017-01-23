/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \file example.cpp

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE example_test

#include <cxtream/core.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/zip.hpp>

#include <cassert>
#include <string>
#include <tuple>
#include <vector>

BOOST_AUTO_TEST_CASE(test_example)
{
    namespace cxs = cxtream::stream;
    using cxs::from; using cxs::to; using cxs::by; using cxs::dim;

    CXTREAM_DEFINE_COLUMN(login, std::string)  // helper macro to define a column of strings
    CXTREAM_DEFINE_COLUMN(age, int)

    std::vector<std::string> logins = {"marry", "ted", "anna", "josh"};
    std::vector<int>           ages = {     24,    41,     16,     59};

    auto stream = ranges::view::zip(logins, ages)
      // create a batched stream out of the raw data
      | cxs::create<login, age>(2)
      // make everyone older by one year
      | cxs::transform(from<age>, to<age>, [](int a) { return a + 1; })
      // increase each letter in the logins by one (i.e., a->b, e->f ...)
      | cxs::transform(from<login>, to<login>, [](char c) { return c + 1; }, dim<2>)
      // increase the ages by the length of the login
      | cxs::transform(from<login, age>, to<age>, [](std::string l, int a) {
            return a + l.length();
        })
      // probabilistically rename 50% of the people to "buzz"
      | cxs::transform(from<login>, to<login>, 0.5, [](std::string) -> std::string {
            return "buzz";
        })
      // drop the login column from the stream
      | cxs::drop<login>
      // introduce the login column back to the stream
      | cxs::transform(from<age>, to<login>, [](int a) {
            return "person_" + std::to_string(a) + "_years_old";
        })
      // filter only people older than 30 years
      | cxs::filter(from<login, age>, by<age>, [](int a) { return a > 30; })
      // asynchronously buffer the stream during iteration
      | cxs::buffer(2);

    // extract the ages from the stream to std::vector
    ages = cxs::unpack(stream, from<age>);
    std::vector<int> desired = {45, 64};
    BOOST_TEST(ages == desired, boost::test_tools::per_element());
    assert((ages == std::vector<int>{45, 64}));
}
