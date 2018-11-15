/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/python/utility/pyboost_is_registered.hpp>

#include <cassert>
#include <vector>

namespace py = boost::python;
namespace hpy = hipipe::python;
// make sure that assert() will be evaluated
#undef NDEBUG

class reg {
};

class not_reg {
};

BOOST_PYTHON_MODULE(pyboost_is_registered_py_cpp)
{
    hpy::initialize();
    py::class_<reg>("reg");

    assert(!hpy::utility::is_registered<double>());
    assert(!hpy::utility::is_registered<std::size_t>());
    assert(hpy::utility::is_registered<reg>());
    assert(!hpy::utility::is_registered<not_reg>());
    assert(!hpy::utility::is_registered<std::vector<int>>());
}
