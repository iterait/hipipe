/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

// The tests for stream::transform are split to multiple
// files to speed up compilation in case of multiple CPUs.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE transform1_test

#include "transform.hpp"


BOOST_AUTO_TEST_CASE(test_partial_transform)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    // Two batches of two columns of a single example.
    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>(3);
    batch1.insert<Unique>(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert<Int>(1);
    batch2.insert<Unique>(std::make_unique<int>(2));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::view::move
      // Increment unique_ptr values by one.
      | hipipe::stream::partial_transform(from<Unique>, to<Unique>,
          [](std::tuple<Unique::batch_type&> data)
            -> std::tuple<Unique::batch_type> {
              BOOST_TEST(std::get<0>(data).size() == 1);
              Unique::batch_type new_data;
              new_data.push_back(std::make_unique<int>(*std::get<0>(data).at(0) + 1));
              return std::make_tuple(std::move(new_data));
        })
      // Put sum of pointer values and ints into ints.
      | hipipe::stream::partial_transform(from<Unique, Int>, to<Int>,
          [](std::tuple<Unique::batch_type&, Int::batch_type> data)
            -> std::tuple<Int::batch_type> {
              BOOST_TEST(std::get<0>(data).size() == 1);
              Int::batch_type new_data;
              new_data.push_back(*std::get<0>(data).at(0) + std::get<1>(data).at(0));
              return std::make_tuple(std::move(new_data));
        })
      // Swap pointer values and ints.
      | hipipe::stream::partial_transform(from<Unique, Int>, to<Int, Unique>,
          [](std::tuple<Unique::batch_type&, Int::batch_type> data)
            -> std::tuple<Int::batch_type, Unique::batch_type> {
              Unique::batch_type new_unique_data;
              new_unique_data.push_back(std::make_unique<int>(std::get<1>(data).at(0)));
              Int::batch_type new_int_data;
              new_int_data.push_back(*std::get<0>(data).at(0));
              return std::make_tuple(std::move(new_int_data), std::move(new_unique_data));
        })
      ;
    
    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Unique>().size() == 1);
    BOOST_TEST(stream.at(0).extract<Int>().size() == 1);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(0) == 9);
    BOOST_TEST(stream.at(0).extract<Int>().at(0) == 6);
    BOOST_TEST(stream.at(1).extract<Unique>().size() == 1);
    BOOST_TEST(stream.at(1).extract<Int>().size() == 1);
    BOOST_TEST(*stream.at(1).extract<Unique>().at(0) == 4);
    BOOST_TEST(stream.at(1).extract<Int>().at(0) == 3);
}


// Transform a single column to itself.
BOOST_AUTO_TEST_CASE(test_to_itself)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>(Int::batch_type{2, 3});
    batch1.insert<Double>(Double::batch_type{4., 5.});
    data.push_back(std::move(batch1));
    batch2.insert<Int>(1);
    batch2.insert<Double>(2.);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::view::move
      | hipipe::stream::transform(from<Int>, to<Int>,
         [](const int& v) { return v - 1; })
      | hipipe::stream::transform(from<Double>, to<Double>,
         [](const int& v) { return v - 1; })
      ;

    BOOST_TEST(stream.size() == 2);
    // batch 0
    BOOST_TEST(stream.at(0).extract<Int>().size()    == 2);
    BOOST_TEST(stream.at(0).extract<Int>().at(0)     == 1);
    BOOST_TEST(stream.at(0).extract<Int>().at(1)     == 2);
    BOOST_TEST(stream.at(0).extract<Double>().size() == 2);
    BOOST_TEST(stream.at(0).extract<Double>().at(0)  == 3.);
    BOOST_TEST(stream.at(0).extract<Double>().at(1)  == 4.);
    // batch 1
    BOOST_TEST(stream.at(1).extract<Int>().size()    == 1);
    BOOST_TEST(stream.at(1).extract<Int>().at(0)     == 0);
    BOOST_TEST(stream.at(1).extract<Double>().size() == 1);
    BOOST_TEST(stream.at(1).extract<Double>().at(0)  == 1.);
}


/*
BOOST_AUTO_TEST_CASE(test_move_only)
{
    // transform move-only column
    std::vector<std::tuple<Int, Unique>> data;
    data.emplace_back(3, std::make_unique<int>(5));
    data.emplace_back(1, std::make_unique<int>(2));

    auto generated = data
      | ranges::view::move
      | transform(from<Unique>, to<Unique, Double>,
          [](const std::unique_ptr<int> &ptr) {
            return std::make_tuple(std::make_unique<int>(*ptr), (double)*ptr);
        })
      | ranges::to_vector;

    // check unique pointers
    std::vector<int> desired_ptr_vals{5, 2};
    for (int i = 0; i < 2; ++i) {
        BOOST_TEST(*(std::get<0>(generated[i]).value()[0]) == desired_ptr_vals[i]);
    }

    // check other
    auto to_check = generated | ranges::view::move | drop<Unique>;
    std::vector<std::tuple<Double, Int>> desired = {{5., 3}, {2., 1}};
    test_ranges_equal(to_check, desired);
}

BOOST_AUTO_TEST_CASE(test_mutable)
{
    std::vector<std::tuple<Int>> data = {{{1, 3}}, {{5, 7}}};

    auto generated = data
      | ranges::view::move
      | transform(from<Int>, to<Int>, [i = 0](const int&) mutable {
            return i++;
        })
      | ranges::to_vector;

    std::vector<std::tuple<Int>> desired = {{{0, 1}}, {{2, 3}}};
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_two_to_one)
{
    // transform two columns to a single column
    std::vector<std::tuple<Int, Double>> data = {{{3, 7}, {5., 1.}}, {1, 2.}};
  
    auto generated = data
      | transform(from<Int, Double>, to<Double>, [](int i, double d) {
            return (double)(i + d);
        });
  
    std::vector<std::tuple<Double, Int>> desired = {{{3 + 5., 7 + 1.}, {3, 7}}, {1 + 2., 1}};
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_one_to_two)
{
    // transform a single column to two columns
    std::vector<std::tuple<Int>> data = {{{3}}, {{1}}};
  
    auto generated = data
      | transform(from<Int>, to<Int, Double>, [](int i) {
            return std::make_tuple(i + i, (double)(i * i));
        });
  
    std::vector<std::tuple<Int, Double>> desired = {{6, 9.}, {2, 1.}};
    test_ranges_equal(generated, desired);
}

BOOST_AUTO_TEST_CASE(test_dim0)
{
    std::vector<std::tuple<Int, Double>> data = {{{3, 2}, 5.}, {1, 2.}};
    auto data_orig = data;

    auto generated = data
      | transform(from<Int>, to<Int>, [](const Int& int_batch) {
            std::vector<int> new_batch = int_batch.value();
            new_batch.push_back(4);
            return new_batch;
        }, dim<0>)
      | ranges::to_vector;

    std::vector<std::tuple<Int, Double>> desired = {{{3, 2, 4}, 5.}, {{1, 4}, 2.}};
    BOOST_CHECK(generated == desired);
    BOOST_CHECK(data == data_orig);
}
*/