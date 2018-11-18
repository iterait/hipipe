/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/python/initialize.hpp>

#include <boost/python.hpp>
#include <opencv2/core/core.hpp>

// check c++ -> python
cv::Point2f point()
{
    cv::Point2f pt = cv::Point2f(4.2, 42);
    return pt;
}

// check python -> c++
bool check_point(cv::Point2f pt)
{
    return std::abs(pt.x - 4.2) < 0.0001 && std::abs(pt.y - 42.0) < 0.0001;
}

BOOST_PYTHON_MODULE(pyboost_cv_point_converter_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    namespace py = boost::python;
    py::def("point", point);
    py::def("check_point", check_point);
}
