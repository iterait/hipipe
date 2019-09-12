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
#define BOOST_TEST_MODULE stream_generate_test

#include "common.hpp"

#include <hipipe/core/stream/generate.hpp>

#include <vector>

HIPIPE_DEFINE_COLUMN(IntVec2d, std::vector<std::vector<int>>)
HIPIPE_DEFINE_COLUMN(Generated, std::vector<int>)
HIPIPE_DEFINE_COLUMN(Generated3d, std::vector<std::vector<std::vector<int>>>)


BOOST_AUTO_TEST_CASE(test_simple)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<IntVec2d>();
    batch1.extract<IntVec2d>().push_back(IntVec2d::example_type{{-1}, {-1}, {}});
    batch1.extract<IntVec2d>().push_back(IntVec2d::example_type{{-1}, {-1},   });
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<IntVec2d>();
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{{-1}, {-1}    });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{              });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{{-1}, {-1}    });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{              });
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | generate(from<IntVec2d>, to<Generated>, [i = 0]() mutable {
            return i++;
        }, 1)
      | generate(from<IntVec2d>, to<Generated3d>, [i = 0]() mutable {
            return std::vector<int>{i++};
        }, 1)
      | ranges::to_vector;

    for (std::size_t i = 0; i < stream.size(); ++i) {
        Generated::data_type generated     = stream.at(i).extract<Generated>();
        Generated3d::data_type generated3d = stream.at(i).extract<Generated3d>();
        switch (i) {
        case 0: BOOST_TEST(generated == (Generated::data_type
                           {{0, 0, 0}, {1, 1}} ));
                BOOST_TEST(generated3d == (Generated3d::data_type
                           {{{{0}}, {{0}}, {}}, {{{1}}, {{1}}}} ));
                break;
        case 1: BOOST_TEST(generated == (Generated::data_type
                           {{2, 2}, {}, {4, 4}, {}} ));
                BOOST_TEST(generated3d == (Generated3d::data_type
                           {{{{2}}, {{2}}}, {}, {{{4}}, {{4}}}, {}} ));
                break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
    }
}
