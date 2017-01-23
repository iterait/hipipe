/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_BUFFER_HPP
#define CXTREAM_CORE_STREAM_BUFFER_HPP

#include <cxtream/core/thread.hpp>

#include <range/v3/core.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>

#include <climits>
#include <deque>
#include <future>

namespace cxtream::stream {

template<typename Rng>
struct buffer_view : ranges::view_facade<buffer_view<Rng>> {
private:
    /// \cond
    friend ranges::range_access;
    /// \endcond

    Rng rng_;
    std::size_t n_;

    struct cursor {
    private:
        buffer_view<Rng>* rng_ = nullptr;
        ranges::iterator_t<Rng> it_ = {};

        std::size_t n_;
        std::deque<std::shared_future<ranges::range_value_type_t<Rng>>> buffer_;

        void pop_buffer()
        {
            if (!buffer_.empty()) {
                buffer_.pop_front();
            }
        }

        void fill_buffer()
        {
            while (it_ != ranges::end(rng_->rng_) && buffer_.size() < n_) {
                auto task = [it = it_]() { return *it; };
                buffer_.emplace_back(global_thread_pool.enqueue(std::move(task)));
                ++it_;
            }
        }

    public:
        cursor() = default;
        explicit cursor(buffer_view<Rng>& rng)
          : rng_{&rng}
          , it_{ranges::begin(rng.rng_)}
          , n_{rng.n_}
        {
            fill_buffer();
        }

        decltype(auto) read() const
        {
            return buffer_.front().get();
        }

        bool equal(ranges::default_sentinel) const
        {
            return buffer_.empty() && it_ == ranges::end(rng_->rng_);
        }

        bool equal(const cursor& that) const
        {
            assert(rng_ == that.rng_);
            return n_ == that.n_ && it_ == that.it_;
        }

        void next()
        {
            pop_buffer();
            fill_buffer();
        }
    };  // class buffer_view

    cursor begin_cursor()
    {
        return cursor{*this};
    }

public:
    buffer_view() = default;

    buffer_view(Rng rng, std::size_t n)
      : rng_{rng}
      , n_{n}
    {
    }

    CONCEPT_REQUIRES(ranges::SizedRange<Rng const>())
    constexpr ranges::range_size_type_t<Rng> size() const
    {
        return ranges::size(rng_);
    }

    CONCEPT_REQUIRES(ranges::SizedRange<Rng>())
    constexpr ranges::range_size_type_t<Rng> size()
    {
        return ranges::size(rng_);
    }
};

class buffer_fn {
private:
    /// \cond
    friend ranges::view::view_access;
    /// \endcond

    static auto bind(buffer_fn buffer, std::size_t n = std::numeric_limits<std::size_t>::max())
    {
        return ranges::make_pipeable(std::bind(buffer, std::placeholders::_1, n));
    }

public:
    template<typename Rng, CONCEPT_REQUIRES_(ranges::ForwardRange<Rng>())>
    buffer_view<ranges::view::all_t<Rng>>
    operator()(Rng&& rng, std::size_t n = std::numeric_limits<std::size_t>::max()) const
    {
        return {ranges::view::all(std::forward<Rng>(rng)), n};
    }
};

/// \ingroup Stream
/// \brief Asynchronously buffers the given range.
///
/// Asynchronously evaluates the given number of elements in advance. When queried for the
/// next element, it is already prepared. This view works for any range, not only
/// for cxtream streams.
///
/// \code
///     std::vector<int> data = {1, 2, 3, 4, 5};
///     auto buffered_rng = data
///       | ranges::view::transform([](int v) { return v + 1; })
///       | buffer(2);
/// \endcode
constexpr ranges::view::view<buffer_fn> buffer{};

}  // end namespace cxtream::stream
#endif
