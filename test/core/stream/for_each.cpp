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
#define BOOST_TEST_MODULE for_each_test

#include "../common.hpp"

#include <hipipe/core/stream/for_each.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/to_container.hpp>


BOOST_AUTO_TEST_CASE(test_for_each_of_two)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>();
    batch1.extract<Int>().push_back(1);
    batch1.extract<Int>().push_back(3);
    batch1.insert<Double>();
    batch1.extract<Double>().push_back(5.);
    batch1.extract<Double>().push_back(6.);
    data.push_back(std::move(batch1));
    batch2.insert<Int>();
    batch2.extract<Int>().push_back(1);
    batch2.insert<Double>();
    batch2.extract<Double>().push_back(2.);
    data.push_back(std::move(batch2));

    // for_each of two columns
    int sum = 0;
    std::vector<batch_t> stream = data
      | ranges::view::move
      | hipipe::stream::for_each(from<Int, Double>,
          [&sum](const int& v, double c) { sum += v; }
        );

    BOOST_TEST(sum == 5);
    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>()    == (std::vector<int>{1, 3}));
    BOOST_TEST(stream.at(0).extract<Double>() == (std::vector<double>{5., 6.}));
    BOOST_TEST(stream.at(1).extract<Int>()    == (std::vector<int>{1}));
    BOOST_TEST(stream.at(1).extract<Double>() == (std::vector<double>{2.}));
}


BOOST_AUTO_TEST_CASE(test_for_each_mutable)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>();
    batch1.extract<Int>().push_back(1);
    batch1.extract<Int>().push_back(3);
    data.push_back(std::move(batch1));
    batch2.insert<Int>();
    batch2.extract<Int>().push_back(5);
    batch2.extract<Int>().push_back(7);
    data.push_back(std::move(batch2));

    struct {
        int i = 0;
        std::shared_ptr<int> i_ptr = std::make_shared<int>(0);
        void operator()(const int&) { *i_ptr = ++i; }
    } func;

    data
      | ranges::view::move
      | hipipe::stream::for_each(from<Int>, func)
      | ranges::to_vector;

    BOOST_TEST(*(func.i_ptr) == 4);
}


BOOST_AUTO_TEST_CASE(test_for_each_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>();
    batch1.extract<Int>().push_back(3);
    batch1.insert<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert<Int>();
    batch2.extract<Int>().push_back(1);
    batch2.insert<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));
  
    std::vector<int> generated;
    data
      | ranges::view::move
      | for_each(from<Int, Unique>,
          [&generated](const int& v, const std::unique_ptr<int>& p) {
              generated.push_back(v + *p);
        })
      | ranges::to_vector;
  
    BOOST_TEST(generated == (std::vector<int>{8, 3}));
}


BOOST_AUTO_TEST_CASE(test_for_each_dim0)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert<Int>();
    batch1.extract<Int>().push_back(3);
    batch1.insert<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert<Int>();
    batch2.extract<Int>().push_back(1);
    batch2.insert<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));
  
    std::vector<int> generated;
    data
      | ranges::view::move
      | hipipe::stream::for_each(from<Int, Unique>,
          [&generated](const std::vector<int>& int_batch,
                       const std::vector<std::unique_ptr<int>>& ptr_batch) {
              generated.push_back(*(ptr_batch[0]) + int_batch[0]);
        }, dim<0>)
      | ranges::to_vector;
  
    BOOST_TEST(generated == (std::vector<int>{8, 3}));
}


BOOST_AUTO_TEST_CASE(test_for_each_dim2_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_move_only_data_2d();

    std::vector<int> generated;
    data
      | ranges::view::move
      | hipipe::stream::for_each(from<UniqueVec>,
          [&generated](std::unique_ptr<int>& ptr) {
              generated.push_back(*ptr + 1);
          }, dim<2>)
      | ranges::to_vector;

    BOOST_TEST(generated == (std::vector<int>{7, 4, 8, 5, 3, 2, 3, 9}));
}


BOOST_AUTO_TEST_CASE(test_for_each_dim2_move_only_mutable)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_move_only_data_2d();

    std::vector<int> generated;
    data
      | ranges::view::move
      | hipipe::stream::for_each(from<UniqueVec>,
          [&generated, i = 4](std::unique_ptr<int>&) mutable {
              generated.push_back(i++);
          }, dim<2>)
      | ranges::to_vector;

    BOOST_TEST(generated == (std::vector<int>{4, 5, 6, 7, 8, 9, 10, 11}));
}
