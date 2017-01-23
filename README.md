# cxtream
[![CircleCI](https://circleci.com/gh/Cognexa/cxtream/tree/master.svg?style=shield)](https://circleci.com/gh/Cognexa/cxtream/tree/master)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-CX%20Experimental-yellow.svg?style=flat)]()
[![Master Developer](https://img.shields.io/badge/master-Filip%20Matzner-lightgrey.svg?style=flat)]()

**This project is under heavy development. The API is continuously changing without regard to backward compatibility.**

__cxtream__ is a C++ library for efficient data processing. Its main purpose is to simplify
and acclelerate data preparation for deep learning models, but it is generic enough to be used
in many other areas.

__cxtream__ lets the programmer build intuitive data streams that transform,
combine and filter the data that pass through. Those streams are compiled,
batched, and asynchronous, therefore maximizing the utilization of the provided
hardware.

- [Documentation and API reference](https://cxtream.org/).
- [Installation guide](https://cxtream.org/installation.html).

## Example

```c++
std::vector<std::string> logins = {"marry", "ted", "anna", "josh"};
std::vector<int>           ages = {     24,    41,     16,     59};

auto stream = ranges::view::zip(logins, ages)

  // create a batched stream out of the raw data
  | cxs::create<login, age>(2)

  // make everyone older by one year
  | cxs::transform(from<age>, to<age>, [](int a) { return a + 1; })

  // increase each letter in the logins by one (i.e., a->b, e->f ...)
  | cxs::transform(from<login>, to<login>, [](char c) { return c + 1; }, dim<2>)

  // increase the ages by the length of the login
  | cxs::transform(from<login, age>, to<age>, [](std::string l, int a) {
        return a + l.length();
    })

  // probabilistically rename 50% of the people to "buzz"
  | cxs::transform(from<login>, to<login>, 0.5, [](std::string) -> std::string {
        return "buzz";
    })

  // drop the login column from the stream
  | cxs::drop<login>

  // introduce the login column back to the stream
  | cxs::transform(from<age>, to<login>, [](int a) {
        return "person_" + std::to_string(a) + "_years_old";
    })

  // filter only people older than 30 years
  | cxs::filter(from<login, age>, by<age>, [](int a) { return a > 30; })

  // asynchronously buffer the stream during iteration
  | cxs::buffer(2);

// extract the ages from the stream to std::vector
ages = cxs::unpack(stream, from<age>);
assert((ages == std::vector<int>{45, 64}));
```
