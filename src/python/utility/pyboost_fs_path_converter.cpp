/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/python/utility/pyboost_fs_path_converter.hpp>

#include <boost/python.hpp>

#include <experimental/filesystem>

namespace cxtream::python::utility {

PyObject* fs_path_to_python_str::convert(const std::experimental::filesystem::path& path)
{
    return boost::python::incref(boost::python::object(path.string()).ptr());
}

fs_path_from_python_str::fs_path_from_python_str()
{
    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<std::experimental::filesystem::path>());
}

void* fs_path_from_python_str::convertible(PyObject* obj_ptr)
{
    if (!PyUnicode_Check(obj_ptr)) return 0;
    return obj_ptr;
}

void fs_path_from_python_str::construct(
  PyObject* obj_ptr,
  boost::python::converter::rvalue_from_python_stage1_data* data)
{
      const char* value = PyUnicode_AsUTF8(obj_ptr);
      if (value == 0) boost::python::throw_error_already_set();
      void* storage = ((boost::python::converter::rvalue_from_python_storage<
                         std::experimental::filesystem::path>*)data)
                        ->storage.bytes;
      new (storage) std::experimental::filesystem::path(value);
      data->convertible = storage;
}

} // namespace cxtream::python::utility
