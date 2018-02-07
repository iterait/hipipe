/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE stream_pad_test

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>
#include <cxtream/core/stream/pad.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

using namespace cxtream::stream;
using namespace cxtream::utility;

CXTREAM_DEFINE_COLUMN(sequences_2d, std::vector<int>)
CXTREAM_DEFINE_COLUMN(sequences_3d, std::list<std::vector<double>>)
CXTREAM_DEFINE_COLUMN(masks_2d, std::vector<bool>)

BOOST_AUTO_TEST_CASE(test_seq_2d_mask_2d)
{
    std::vector<std::vector<int>> data = {{1, 2}, {3, 4, 5}, {}, {6, 7}};
    auto stream = data
      | create<sequences_2d>(2)
      | pad(from<sequences_2d>, mask<masks_2d>, -1);

    int batch_i = 0;
    for (auto batch : stream) {
        auto seqs = std::get<sequences_2d>(batch).value();
        auto mask = std::get<masks_2d>(batch).value();

        // check the contents
        switch (batch_i) {
        case 0: BOOST_CHECK((seqs ==
                  std::vector<std::vector<int>>{{1, 2, -1}, {3, 4, 5}}));
                BOOST_CHECK((mask ==
                  std::vector<std::vector<bool>>{{true, true, false}, {true, true, true}}));
                break;
        case 1: BOOST_CHECK((seqs ==
                  std::vector<std::vector<int>>{{-1, -1}, {6, 7}}));
                BOOST_CHECK((mask ==
                  std::vector<std::vector<bool>>{{false, false}, {true, true}}));
                break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
        ++batch_i;
    }
}

BOOST_AUTO_TEST_CASE(test_seq_3d_mask_2d)
{
    std::vector<std::list<std::vector<double>>> data =
      {{{1.}, {2., 3.}}, {{3., 4.}}, {}, {{5., 1}}};
    auto stream = data
      | create<sequences_3d>(2)
      | pad(from<sequences_3d>, mask<masks_2d>, {-1., -1.});

    int batch_i = 0;
    for (auto batch : stream) {
        auto seqs = std::get<sequences_3d>(batch).value();
        auto mask = std::get<masks_2d>(batch).value();

        // check the contents
        switch (batch_i) {
        case 0: BOOST_CHECK((seqs ==
                  std::vector<std::list<std::vector<double>>>
                    {{{1.}, {2., 3.}}, {{3., 4.}, {-1., -1.}}}));
                BOOST_CHECK((mask ==
                  std::vector<std::vector<bool>>{{true, true}, {true, false}}));
                break;
        case 1: BOOST_CHECK((seqs ==
                  std::vector<std::list<std::vector<double>>>
                    {{{-1., -1.}}, {{5., 1.}}}));
                BOOST_CHECK((mask ==
                  std::vector<std::vector<bool>>{{false}, {true}}));
                break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
        ++batch_i;
    }
}
