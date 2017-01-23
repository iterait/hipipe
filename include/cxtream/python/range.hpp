/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_RANGE_HPP
#define CXTREAM_PYTHON_RANGE_HPP

#include <cxtream/python/utility/pyboost_is_registered.hpp>

#include <boost/python.hpp>
#include <range/v3/core.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cxtream::python {

/// \ingroup Python
/// \brief Exception which is translated to Python's StopIteration when thrown.
struct stop_iteration_exception : public std::runtime_error {
    stop_iteration_exception()
      : std::runtime_error{"stop iteration"}
    { }
};

/// boost::python translation function for the stop_iteration_exception.
void stop_iteration_translator(const stop_iteration_exception& x)
{
    PyErr_SetNone(PyExc_StopIteration);
}

/// \ingroup Python
/// \brief Python adapter for C++ ranges.
///
/// This class provides `__next__`, `__iter__`, `__len__`, and `__getitem__` methods
/// emulating python containers. `__getitem__` and `__len__` are provided only for
/// random access ranges.
///
/// Note that this class is useful for infinite ranges and views.
/// However, for normal finite ranges, it seems to be many times slower
/// than using boost::python::list.
///
/// Example:
/// \code
///     std::vector<int> data = {1, 2, 3};
///     range<std::vector<int>> py_range{std::move(data)};
///     // py_range is now a registered boost::python object
///     // supporting iteration and random access.
/// \endcode
template<typename Rng>
class range {
private:
    std::shared_ptr<Rng> rng_ptr_;

    // register __len__ function if it is supported
    CONCEPT_REQUIRES(ranges::SizedRange<const Rng>())
    static void register_len(boost::python::class_<range<Rng>>& cls)
    {
        cls.def("__len__", &range<Rng>::len<>);
    }
    CONCEPT_REQUIRES(!ranges::SizedRange<const Rng>())
    static void register_len(boost::python::class_<range<Rng>>&)
    {
    }

    // register __getitem__ function if it is supported
    CONCEPT_REQUIRES(ranges::RandomAccessRange<const Rng>())
    static void register_getitem(boost::python::class_<range<Rng>>& cls)
    {
        cls.def("__getitem__", &range<Rng>::getitem<>);
    }
    CONCEPT_REQUIRES(!ranges::RandomAccessRange<const Rng>())
    static void register_getitem(boost::python::class_<range<Rng>>&)
    {
    }

    // function to register the type of this class in boost::python
    // makes sure the type is registered only once
    static void register_to_python()
    {
        namespace py = boost::python;

        if (!utility::is_registered<range<Rng>>()) {
            std::string this_name = std::string("cxtream_") + typeid(range<Rng>).name();
            py::class_<range<Rng>> cls{this_name.c_str(), py::no_init};
            cls.def("__iter__", &range<Rng>::iter);
            register_len(cls);
            register_getitem(cls);

            py::class_<range<Rng>::iterator>{(this_name + "_iterator").c_str(), py::no_init}
              .def("__next__", &range<Rng>::iterator::next)
              .def("__iter__", &range<Rng>::iterator::iter);
        };
    }

public:
    class iterator {
    private:
        std::shared_ptr<Rng> rng_ptr_;
        ranges::iterator_t<Rng> position_;
        bool first_iteration_;

    public:
        iterator() = default;

        // Construct iterator from a range.
        explicit iterator(range& rng)
          : rng_ptr_{rng.rng_ptr_},
            position_{ranges::begin(*rng_ptr_)},
            first_iteration_{true}
        {
        }

        // Return a copy of this iterator.
        iterator iter()
        {
            return *this;
        }

        // Return the next element in the range.
        //
        // \throws stop_iteration_exception if there are no more elements.
        auto next()
        {
            // do not increment the iterator in the first iteration, just return *begin()
            if (!first_iteration_ && position_ != ranges::end(*rng_ptr_)) ++position_;
            first_iteration_ = false;
            if (position_ == ranges::end(*rng_ptr_)) throw stop_iteration_exception();
            return *position_;
        }
    };

    /// Default empty constructor.
    range() = default;

    /// Construct python range from a C++ range. This function creates a copy of the range.
    explicit range(Rng rng)
      : rng_ptr_{std::make_shared<Rng>(std::move(rng))}
    {
        register_to_python();
    }

    /// Construct python range from a std::shared_ptr to a C++ range.
    range(std::shared_ptr<Rng> rng_ptr)
      : rng_ptr_{std::move(rng_ptr)}
    {
        register_to_python();
    }

    // Get python iterator.
    iterator iter()
    {
        return iterator{*this};
    }

    // Get an item or a slice.
    //
    // Note that when slicing, the data get copied.
    CONCEPT_REQUIRES(ranges::RandomAccessRange<const Rng>())
    boost::python::object getitem(PyObject* idx_py) const
    {
        // handle slices
        if (PySlice_Check(idx_py)) {
            PySliceObject* slice = static_cast<PySliceObject*>(static_cast<void*>(idx_py));
            if (slice->step != Py_None) {
                throw std::logic_error("Cxtream python range does not support slice steps.");
            }

            auto handle_index = [this](PyObject* idx_py, long def_val) {
                long idx = def_val;
                if (idx_py != Py_None) {
                    idx = boost::python::extract<long>(idx_py);
                    // reverse negative index
                    if (idx < 0) idx += this->len();
                    // if it is still negative, clip
                    if (idx < 0) idx = 0;
                    // handle index larger then len
                    if (idx > this->len()) idx = len();
                }
                return idx;
            };
            long start = handle_index(slice->start, 0);
            long stop = handle_index(slice->stop, len());
            if (start > stop) start = stop;

            using slice_data_type = std::vector<ranges::range_value_type_t<Rng>>;
            slice_data_type slice_data{rng_ptr_->begin() + start, rng_ptr_->begin() + stop};
            return boost::python::object{range<slice_data_type>{std::move(slice_data)}};
        }

        // handle indices
        long idx = boost::python::extract<long>(idx_py);
        if (idx < 0) idx += len();
        return boost::python::object{ranges::at(*rng_ptr_, idx)};
    }

    // Get the size of the range.
    CONCEPT_REQUIRES(ranges::SizedRange<const Rng>())
    long len() const
    {
        return ranges::size(*rng_ptr_);
    }

};  // class range

}  // end namespace cxtream::python
#endif
