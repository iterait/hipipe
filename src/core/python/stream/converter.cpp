/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/build_config.hpp>
#ifdef HIPIPE_BUILD_PYTHON

#include <hipipe/core/python/range.hpp>
#include <hipipe/core/python/stream/converter.hpp>

#include <range/v3/view/transform.hpp>

namespace hipipe::python::stream {

template<typename CONTAINER>
class owning_iterator : public decltype(CONTAINER().begin()) {
private:
    std::shared_ptr<CONTAINER> container;

public:
    owning_iterator() = default;
    owning_iterator(std::shared_ptr<CONTAINER> c) : decltype(CONTAINER().begin())(c->begin()), container(c) { };
};


__attribute__((visibility("default"))) pybind11::object to_python(hipipe::stream::input_stream_t stream) {
    ranges::any_view<pybind11::dict> range_of_dicts =
      ranges::view::transform(std::move(stream), &hipipe::stream::batch_t::to_python);
    
    std::shared_ptr<ranges::any_view<pybind11::dict>> range_ptr = std::make_shared<ranges::any_view<pybind11::dict>>(range_of_dicts);
    owning_iterator<ranges::any_view<pybind11::dict>> begin(range_ptr);

    return pybind11::make_iterator(begin, range_ptr->end());
}

}  // end namespace hipipe::python::stream

#endif  // HIPIPE_BUILD_PYTHON
