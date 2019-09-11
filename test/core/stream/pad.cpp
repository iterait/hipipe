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
#define BOOST_TEST_MODULE stream_pad_test

#include "common.hpp"

#include <hipipe/core/stream/pad.hpp>


HIPIPE_DEFINE_COLUMN(Sequences, std::vector<int>)
HIPIPE_DEFINE_COLUMN(Sequences2d, std::list<std::vector<double>>)
HIPIPE_DEFINE_COLUMN(Mask, std::vector<bool>)


BOOST_AUTO_TEST_CASE(test_sequences_mask)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::mask;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Sequences>();
    batch1.extract<Sequences>().push_back(Sequences::example_type{1, 2   });
    batch1.extract<Sequences>().push_back(Sequences::example_type{3, 4, 5});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Sequences>();
    batch2.extract<Sequences>().push_back(Sequences::example_type{       });
    batch2.extract<Sequences>().push_back(Sequences::example_type{6, 7   });
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::pad(from<Sequences>, mask<Mask>, -1)
      | ranges::to_vector;

    for (std::size_t i = 0; i < stream.size(); ++i) {
        Sequences::data_type seqs = stream.at(i).extract<Sequences>();
        Mask::data_type     mask = stream.at(i).extract<Mask>();
        switch (i) {
        case 0: BOOST_TEST(seqs ==
                  (std::vector<std::vector<int>>{{1, 2, -1}, {3, 4, 5}}));
                BOOST_TEST(mask ==
                  (std::vector<std::vector<bool>>{{true, true, false}, {true, true, true}}));
                break;
        case 1: BOOST_TEST(seqs ==
                  (std::vector<std::vector<int>>{{-1, -1}, {6, 7}}));
                BOOST_TEST(mask ==
                  (std::vector<std::vector<bool>>{{false, false}, {true, true}}));
                break;
        default: BOOST_FAIL("Only two batches should be provided.");
        }
    }
}


BOOST_AUTO_TEST_CASE(test_sequences_2d_mask)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::mask;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Sequences2d>();
    batch1.extract<Sequences2d>().push_back(Sequences2d::example_type{{1.}, {2., 3.}});
    batch1.extract<Sequences2d>().push_back(Sequences2d::example_type{{3., 4.}});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Sequences2d>();
    batch2.extract<Sequences2d>().push_back(Sequences2d::example_type{        });
    batch2.extract<Sequences2d>().push_back(Sequences2d::example_type{{5., 1.}});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::pad(from<Sequences2d>, mask<Mask>, {-1., -1.})
      | ranges::to_vector;

    for (std::size_t i = 0; i < stream.size(); ++i) {
        Sequences2d::data_type seqs = stream.at(i).extract<Sequences2d>();
        Mask::data_type        mask = stream.at(i).extract<Mask>();
        switch (i) {
        case 0: BOOST_TEST(seqs ==
                  (std::vector<std::list<std::vector<double>>>
                    {{{1.}, {2., 3.}}, {{3., 4.}, {-1., -1.}}}));
                BOOST_TEST(mask ==
                  (std::vector<std::vector<bool>>{{true, true}, {true, false}}));
                break;
        case 1: BOOST_TEST(seqs ==
                  (std::vector<std::list<std::vector<double>>>
                    {{{-1., -1.}}, {{5., 1.}}}));
                BOOST_TEST(mask ==
                  (std::vector<std::vector<bool>>{{false}, {true}}));
                break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
    }
}
