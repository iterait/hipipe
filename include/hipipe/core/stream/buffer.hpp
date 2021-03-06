/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef HIPIPE_CORE_STREAM_BUFFER_HPP
#define HIPIPE_CORE_STREAM_BUFFER_HPP

#include <hipipe/core/thread.hpp>

#include <range/v3/core.hpp>
#include <range/v3/functional/bind_back.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>

#include <climits>
#include <deque>
#include <future>
#include <memory>

namespace hipipe::stream {

namespace rg = ranges;
namespace rgv = ranges::views;

template<typename Rng>
struct buffer_view : rg::view_facade<buffer_view<Rng>> {
private:
    /// \cond
    friend rg::range_access;
    /// \endcond

    Rng rng_;
    std::size_t n_;

    struct cursor {
    private:
        buffer_view<Rng>* rng_ = nullptr;
        rg::iterator_t<Rng> it_ = {};
        using value_type = rg::range_value_t<Rng>;
        using reference_type = value_type&;

        std::size_t n_;

        // std::shared_future only allows retrieving the shared state via
        // a const reference. Therefore, we store the computed results
        // on heap (shared_ptr) and return references to those objects
        // (non-const references).
        std::deque<std::shared_future<std::shared_ptr<value_type>>> buffer_;

        void pop_buffer()
        {
            if (!buffer_.empty()) {
                buffer_.pop_front();
            }
        }

        void fill_buffer()
        {
            while (it_ != rg::end(rng_->rng_) && buffer_.size() < n_) {
                auto task = [it = it_]() { return std::make_shared<value_type>(*it); };
                buffer_.emplace_back(global_thread_pool.enqueue(std::move(task)));
                ++it_;
            }
        }

    public:
        using single_pass = std::true_type;

        cursor() = default;

        explicit cursor(buffer_view<Rng>& rng)
          : rng_{&rng}
          , it_{rg::begin(rng.rng_)}
          , n_{rng.n_}
        {
            fill_buffer();
        }

        value_type&& read() const
        {
            return std::move(*buffer_.front().get());
        }

        bool equal(rg::default_sentinel_t) const
        {
            return buffer_.empty() && it_ == rg::end(rng_->rng_);
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

        ~cursor()
        {
            for (auto& future : buffer_) {
                if (future.valid()) future.wait();
            }
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

    CPP_template(int dummy = 0)(requires rg::sized_range<const Rng>)
    constexpr rg::range_size_type_t<Rng> size() const
    {
        return rg::size(rng_);
    }

    CPP_template(int dummy = 0)(requires rg::sized_range<const Rng>)
    constexpr rg::range_size_type_t<Rng> size()
    {
        return rg::size(rng_);
    }
};

class buffer_fn {
public:
    CPP_template(typename Rng)(requires rg::forward_range<Rng>)
    buffer_view<rgv::all_t<Rng>>
    operator()(Rng&& rng, std::size_t n = std::numeric_limits<std::size_t>::max()) const
    {
        return {rgv::all(std::forward<Rng>(rng)), n};
    }

    auto operator()(std::size_t n = std::numeric_limits<std::size_t>::max()) const
    {
        return rg::make_view_closure(rg::bind_back(buffer_fn{}, n));
    }
};

/// \ingroup Stream
/// \brief Asynchronously buffers the given range.
///
/// Asynchronously evaluates the given number of elements in advance. When queried for the
/// next element, it is already prepared. This view works for any range, not only
/// for hipipe streams.
///
/// Note that this transformer is not lazy and instead _eagerly evaluates_ the
/// data in asynchronous threads. To avoid recalculation of the entire underlying
/// range whenever e.g., std::distance is called, this transformer intentionally
/// changes the stream type to input_range. The downside is that no further
/// transformations can be appended (except for \ref Stream stream::rebatch) and everything
/// has to be prepared before the application of this transformer.
///
/// \code
///     std::vector<int> data = {1, 2, 3, 4, 5};
///     auto buffered_rng = data
///       | rgv::transform([](int v) { return v + 1; })
///       | buffer(2);
/// \endcode
inline rgv::view_closure<buffer_fn> buffer{};

}  // end namespace hipipe::stream
#endif
