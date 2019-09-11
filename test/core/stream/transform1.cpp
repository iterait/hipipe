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

#include "common.hpp"

#include <hipipe/core/stream/transform.hpp>


BOOST_AUTO_TEST_CASE(test_partial_transform)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    // Two batches of two columns of a single example.
    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      // Increment unique_ptr values by one.
      | hipipe::stream::partial_transform(from<Unique>, to<Unique>,
          [](std::tuple<Unique::data_type&> data)
            -> std::tuple<Unique::data_type> {
              BOOST_TEST(std::get<0>(data).size() == 1);
              Unique::data_type new_data;
              new_data.push_back(std::make_unique<int>(*std::get<0>(data).at(0) + 1));
              return std::make_tuple(std::move(new_data));
        })
      // Put sum of pointer values and ints into ints.
      | hipipe::stream::partial_transform(from<Unique, Int>, to<Int>,
          [](std::tuple<Unique::data_type&, Int::data_type> data)
            -> std::tuple<Int::data_type> {
              BOOST_TEST(std::get<0>(data).size() == 1);
              Int::data_type new_data;
              new_data.push_back(*std::get<0>(data).at(0) + std::get<1>(data).at(0));
              return std::make_tuple(std::move(new_data));
        })
      // Swap pointer values and ints.
      | hipipe::stream::partial_transform(from<Unique, Int>, to<Int, Unique>,
          [](std::tuple<Unique::data_type&, Int::data_type> data)
            -> std::tuple<Int::data_type, Unique::data_type> {
              Unique::data_type new_unique_data;
              new_unique_data.push_back(std::make_unique<int>(std::get<1>(data).at(0)));
              Int::data_type new_int_data;
              new_int_data.push_back(*std::get<0>(data).at(0));
              return std::make_tuple(std::move(new_int_data), std::move(new_unique_data));
        })
      | ranges::to_vector;

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
    batch1.insert_or_assign<Int>(Int::data_type{2, 3});
    batch1.insert_or_assign<Double>(Double::data_type{4., 5.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    batch2.insert_or_assign<Double>(2.);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::transform(from<Int>, to<Int>,
         [](const int& v) { return v - 1; })
      | hipipe::stream::transform(from<Double>, to<Double>,
         [](const int& v) { return v - 1; })
      | ranges::to_vector;

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


// Transform a move-only column.
BOOST_AUTO_TEST_CASE(test_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::transform(from<Unique>, to<Unique>,
          [](const std::unique_ptr<int> &ptr) {
              return std::make_unique<int>(*ptr + 1);
        })
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(0) == 6);
    BOOST_TEST(stream.at(1).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(1).extract<Unique>().at(0) == 3);
}


// Test mutable transformation function.
BOOST_AUTO_TEST_CASE(test_mutable)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::data_type{1, 5, 3});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::data_type{3, 5});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | transform(from<Int>, to<Int>, [i = 0](const int&) mutable {
            return i++;
        })
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>() == (std::vector<int>{0, 1, 2}));
    BOOST_TEST(stream.at(1).extract<Int>() == (std::vector<int>{3, 4}));
}


// Two columns to one column.
BOOST_AUTO_TEST_CASE(test_two_to_one)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::data_type{3, 7});
    batch1.insert_or_assign<Double>(Double::data_type{5., 1.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::data_type{1, 2});
    batch2.insert_or_assign<Double>(Double::data_type{3., 7.});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | transform(from<Int, Double>, to<Double>, [](int i, double d) -> double {
            return i + d;
        })
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>()    == (std::vector<int>   {3, 7}));
    BOOST_TEST(stream.at(0).extract<Double>() == (std::vector<double>{8., 8.}));
    BOOST_TEST(stream.at(1).extract<Int>()    == (std::vector<int>   {1, 2}));
    BOOST_TEST(stream.at(1).extract<Double>() == (std::vector<double>{4., 9.}));
}


// One column to two columns.
BOOST_AUTO_TEST_CASE(test_one_to_two)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::data_type{3, 7});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::data_type{1, 2});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | transform(from<Int>, to<Int, Double>, [](int i) {
            return std::make_tuple(i + i, (double)(i * i));
        })
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>()    == (std::vector<int>   {6,  14 }));
    BOOST_TEST(stream.at(0).extract<Double>() == (std::vector<double>{9., 49.}));
    BOOST_TEST(stream.at(1).extract<Int>()    == (std::vector<int>   {2,  4  }));
    BOOST_TEST(stream.at(1).extract<Double>() == (std::vector<double>{1., 4. }));
}


// Test transformation of whole batches.
BOOST_AUTO_TEST_CASE(test_dim0)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::data_type{3, 7});
    batch1.insert_or_assign<Double>(Double::data_type{2.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::data_type{1, 2});
    batch2.insert_or_assign<Double>(Double::data_type{6.});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | transform(from<Int>, to<Int>, [](const Int::data_type& int_batch) {
            std::vector<int> new_batch = int_batch;
            new_batch.push_back(4);
            return new_batch;
        }, dim<0>)
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>()    == (std::vector<int>   {3, 7, 4}));
    BOOST_TEST(stream.at(0).extract<Double>() == (std::vector<double>{2.     }));
    BOOST_TEST(stream.at(1).extract<Int>()    == (std::vector<int>   {1, 2, 4}));
    BOOST_TEST(stream.at(1).extract<Double>() == (std::vector<double>{6.     }));
}
