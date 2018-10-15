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

#include <hipipe/core/stream/column.hpp>

#include <range/v3/core.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>

#include <algorithm>

namespace hipipe::stream {


template <typename Rng>
struct batch_view : ranges::view_facade<batch_view<Rng>> {
private:
    /// \cond
    friend ranges::range_access;
    /// \endcond
    Rng rng_;
    std::size_t n_;

    struct cursor {
    private:
        batch_view<Rng>* rng_ = nullptr;
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

        explicit cursor(batch_view<Rng>& rng)
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

        bool equal(ranges::default_sentinel) const
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
    batch_view() = default;
    batch_view(Rng rng, std::size_t n)
      : rng_{rng}
      , n_{n}
    {
        if (n_ <= 0) {
            throw std::invalid_argument{"hipipe::stream::batch:"
              " The new batch size " + std::to_string(n_) + " is not strictly positive."};
        }
    }
};  // class batch_view

class batch_fn {
private:
    /// \cond
    friend ranges::view::view_access;
    /// \endcond

    static auto bind(batch_fn batch, std::size_t n)
    {
        return ranges::make_pipeable(std::bind(batch, std::placeholders::_1, n));
    }

public:
    template <typename Rng, CONCEPT_REQUIRES_(ranges::InputRange<Rng>())>
    batch_view<ranges::view::all_t<Rng>> operator()(Rng&& rng, std::size_t n) const
    {
        return {ranges::view::all(std::forward<Rng>(rng)), n};
    }
};  // class batch_fn


/// \ingroup Stream
/// \brief Accumulate the stream and yield batches of a different size.
///
/// The batch size of the accumulated columns is allowed to differ between batches.
/// To make one large batch of all the data, use std::numeric_limits<std::size_t>::max().
///
/// Note that this object evaluates the batches computed by the stream and builds
/// batches of a different size. Therefore, if you want to combine this object with
/// e.g. \ref stream::buffer, the buffer needs to preceed \ref stream::batch for it to have
/// any effect.
///
/// \code
///     HIPIPE_DEFINE_COLUMN(value, int)
///     auto rng = view::iota(0, 10)
///       | create<value>(2)  // batches the data by two examples
///       | batch(3);         // changes the batch size to three examples
/// \endcode
inline ranges::view::view<batch_fn> batch{};

}  // namespace hipipe::stream
