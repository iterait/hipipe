/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/core/stream/stream_t.hpp>

#include <range/v3/core.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>

#include <algorithm>

namespace hipipe::stream {


template <typename Rng>
struct rebatch_view : ranges::view_facade<rebatch_view<Rng>> {
private:
    /// \cond
    friend ranges::range_access;
    /// \endcond
    Rng rng_;
    std::size_t n_;

    struct cursor {
    private:
        rebatch_view<Rng>* rng_ = nullptr;
        ranges::iterator_t<Rng> it_ = {};

        // the batch into which we accumulate the data
        // the batch will be a pointer to allow moving from it in const functions
        std::shared_ptr<batch_t> batch_;

        // the subbatch of the original range
        std::shared_ptr<batch_t> subbatch_;

        // whether the underlying range is at the end of iteration
        bool done_ = false;

        // find the first non-empty subbatch and return if successful
        bool find_next()
        {
            while (subbatch_->batch_size() == 0) {
                if (it_ == ranges::end(rng_->rng_) || ++it_ == ranges::end(rng_->rng_)) {
                    return false;
                }
                subbatch_ = std::make_shared<batch_t>(*it_);
            }
            return true;
        }

        // fill the batch_ with the elements from the current subbatch_
        void fill_batch()
        {
            do {
                assert(batch_->batch_size() < rng_->n_);
                std::size_t to_take =
                  std::min(rng_->n_ - batch_->batch_size(), subbatch_->batch_size());
                batch_->push_back(subbatch_->take(to_take));
            } while (batch_->batch_size() < rng_->n_ && find_next());
        }

    public:
        using single_pass = std::true_type;

        cursor() = default;

        explicit cursor(rebatch_view<Rng>& rng)
          : rng_{&rng}
          , it_{ranges::begin(rng_->rng_)}
        {
            // do nothing if the subrange is empty
            if (it_ == ranges::end(rng_->rng_)) {
                done_ = true;
            } else {
                subbatch_ = std::make_shared<batch_t>(*it_);
                next();
            }
        }

        batch_t&& read() const
        {
            return std::move(*batch_);
        }

        bool equal(ranges::default_sentinel_t) const
        {
            return done_;
        }

        bool equal(const cursor& that) const
        {
            assert(rng_ == that.rng_);
            return it_ == that.it_ && subbatch_->batch_size() == that.subbatch_->batch_size();
        }

        void next()
        {
            batch_ = std::make_shared<batch_t>();
            if (find_next()) fill_batch();
            else done_ = true;
        }
    };  // struct cursor

    cursor begin_cursor() { return cursor{*this}; }

public:
    rebatch_view() = default;
    rebatch_view(Rng rng, std::size_t n)
      : rng_{rng}
      , n_{n}
    {
        if (n_ <= 0) {
            throw std::invalid_argument{"hipipe::stream::rebatch:"
              " The new batch size " + std::to_string(n_) + " is not strictly positive."};
        }
    }
};  // class rebatch_view

class rebatch_fn {
private:
    /// \cond
    friend ranges::view::view_access;
    /// \endcond

    static auto bind(rebatch_fn rebatch, std::size_t n)
    {
        return ranges::make_pipeable(std::bind(rebatch, std::placeholders::_1, n));
    }

public:
    CPP_template(class Rng)(requires ranges::InputRange<Rng>)
    rebatch_view<ranges::view::all_t<Rng>> operator()(Rng&& rng, std::size_t n) const
    {
        return {ranges::view::all(std::forward<Rng>(rng)), n};
    }
};  // class rebatch_fn


/// \ingroup Stream
/// \brief Accumulate the stream and yield batches of a different size.
///
/// The batch size of the accumulated columns is allowed to differ between batches.
/// To make one large batch of all the data, use std::numeric_limits<std::size_t>::max().
///
/// Note that this stream transformer is not lazy and instead _eagerly evaluates_
/// the batches computed by the previous stream pipeline and reorganizes the
/// evaluated data to batches of a different size. To avoid recalculation of the
/// entire stream whenever e.g., std::distance is called, this transformer
/// intentionally changes the stream type to InputRange. The downside is that no
/// further transformations or buffering can be appended and everything has to be
/// prepared before the application of this transformer.
///
/// \code
///     HIPIPE_DEFINE_COLUMN(value, int)
///     auto rng = view::iota(0, 10)
///       | create<value>(2)  // batches the data by two examples
///       | rebatch(3);       // changes the batch size to three examples
/// \endcode
inline ranges::view::view<rebatch_fn> rebatch{};

}  // namespace hipipe::stream
