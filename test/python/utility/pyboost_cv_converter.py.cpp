/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/python/initialize.hpp>

#include <boost/python.hpp>
#include <opencv2/core/core.hpp>

#include <algorithm>
#include <vector>

// chessboard matrix //

// check c++ -> python
cv::Mat chessboard()
{
    std::vector<std::uint8_t> data = {
      1, 0, 1,
      0, 1, 0,
      1, 0, 1,
      0, 1, 0};
    cv::Mat mat(4, 3, CV_8UC1);
    memcpy(mat.data, data.data(), data.size());
    return mat;
}

// check python -> c++
bool check_chessboard(cv::Mat mat)
{
    return cv::countNonZero(mat != chessboard()) == 0;
}

// 3-channeled rgb matrix //

// check c++ -> python
cv::Mat rgb_sample()
{
    std::vector<float> data = {
      0.1, 0.2, 0.3,
      0.1, 0.2, 0.3,
      0.4, 0.5, 0.6,
      0.4, 0.5, 0.6,
      0.7, 0.8, 0.9,
      0.7, 0.8, 0.9};
    cv::Mat mat(2, 3, CV_32FC3);
    memcpy(mat.data, data.data(), data.size() * sizeof(float));
    return mat;
}

// check python -> c++
bool check_rgb_sample(cv::Mat mat)
{
    // since OpenCV 2 cannot compare matrices with more than one channel,
    // we will use std::equal instead
    cv::Mat gold = rgb_sample();
    return std::equal(mat.begin<float>(), mat.end<float>(), gold.begin<float>());
}

BOOST_PYTHON_MODULE(pyboost_cv_converter_py_cpp)
{
    // initialize cxtream OpenCV converters, exceptions, etc.
    cxtream::python::initialize();

    // expose the functions
    namespace py = boost::python;
    py::def("chessboard", chessboard);
    py::def("check_chessboard", check_chessboard);
    py::def("rgb_sample", rgb_sample);
    py::def("check_rgb_sample", check_rgb_sample);
}
