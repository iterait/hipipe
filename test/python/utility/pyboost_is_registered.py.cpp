/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/python/initialize.hpp>
#include <cxtream/python/utility/pyboost_is_registered.hpp>

#include <cassert>
#include <vector>

namespace py = boost::python;
namespace cxpy = cxtream::python;
// make sure that assert() will be evaluated
#undef NDEBUG

class reg {
};

class not_reg {
};

BOOST_PYTHON_MODULE(pyboost_is_registered_py_cpp)
{
    cxpy::initialize();
    py::class_<reg>("reg");

    assert(!cxpy::utility::is_registered<double>());
    assert(!cxpy::utility::is_registered<std::size_t>());
    assert(cxpy::utility::is_registered<reg>());
    assert(!cxpy::utility::is_registered<not_reg>());
    assert(!cxpy::utility::is_registered<std::vector<int>>());
}
