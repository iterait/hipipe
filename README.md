# HiPipe
[![CircleCI](https://circleci.com/gh/iterait/hipipe/tree/dev.svg?style=shield)](https://circleci.com/gh/iterait/hipipe/tree/dev)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-Regular-brightgreen.svg?style=flat)]()
[![Master Developer](https://img.shields.io/badge/master-Filip%20Matzner-lightgrey.svg?style=flat)]()


__HiPipe__ is a C++ library for efficient data processing. Its main purpose is to simplify
and accelerate data preparation for deep learning models, but it is generic enough to be used
in many other areas.

__HiPipe__ lets the programmer build intuitive data streams that transform,
combine and filter the data that pass through. Those streams are compiled,
batched, and asynchronous, therefore maximizing the utilization of the provided
hardware.

- [Documentation and API reference](https://hipipe.org/).
- [Installation guide](https://hipipe.org/installation.html).

## Example

```c++
std::vector<std::string> logins = {"marry", "ted", "anna", "josh"};
std::vector<int>           ages = {     24,    41,     16,     59};

auto stream = ranges::views::zip(logins, ages)

  // create a batched stream out of the raw data
  | hipipe::create<login, age>(2)

  // make everyone older by one year
  | hipipe::transform(from<age>, to<age>, [](int a) { return a + 1; })

  // increase each letter in the logins by one (i.e., a->b, e->f ...)
  | hipipe::transform(from<login>, to<login>, [](char c) { return c + 1; }, dim<2>)

  // increase the ages by the length of the login
  | hipipe::transform(from<login, age>, to<age>, [](std::string l, int a) {
        return a + l.length();
    })

  // probabilistically rename 50% of the people to "buzz"
  | hipipe::transform(from<login>, to<login>, 0.5, [](std::string) -> std::string {
        return "buzz";
    })

  // drop the login column from the stream
  | hipipe::drop<login>

  // introduce the login column back to the stream
  | hipipe::transform(from<age>, to<login>, [](int a) {
        return "person_" + std::to_string(a) + "_years_old";
    })

  // filter only people older than 30 years
  | hipipe::filter(from<login, age>, by<age>, [](int a) { return a > 30; })

  // asynchronously buffer the stream during iteration
  | hipipe::buffer(2);

// extract the ages from the stream to std::vector
ages = hipipe::unpack(stream, from<age>);
assert((ages == std::vector<int>{45, 64}));
```
