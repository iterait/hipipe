/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_BATCH_HPP
#define CXTREAM_CORE_STREAM_BATCH_HPP

#include <cxtream/core/utility/tuple.hpp>

#include <range/v3/core.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>

#include <algorithm>

namespace cxtream::stream {

/// Check whether all columns in a tuple have the same batch size.
template<typename Tuple>
constexpr bool is_same_batch_size(const Tuple& tuple)
{
    bool same = true;
    if (std::tuple_size<Tuple>{} > 0) {
        auto bs = std::get<0>(tuple).value().size();
        utility::tuple_for_each(tuple, [bs, &same](const auto& column) {
            same &= (column.value().size() == bs);
        });
    }
    return same;
}

/// Get the batch size of a column tuple.
template<typename Tuple>
constexpr std::size_t batch_size(const Tuple& tuple)
{
    static_assert(std::tuple_size<Tuple>{} && "Cannot get batch size if there are no columns");
    assert(is_same_batch_size(tuple) && "All the columns have to have equal batch size");
    return std::get<0>(tuple).value().size();
}

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

        using batch_t_ = ranges::range_value_type_t<Rng>;
        using column_idxs_t = std::make_index_sequence<std::tuple_size<batch_t_>{}>;
        // the batch into which we accumulate the data
        // the batch will be a pointer to allow moving from it in const functions
        std::shared_ptr<batch_t_> batch_ = std::make_shared<batch_t_>();

        // the subbatch of the original range
        std::shared_ptr<batch_t_> subbatch_;
        // the current index into the batch of the original range
        std::size_t subbatch_idx_ = 0;

        bool done_ = false;

        // reserve space in subbatch
        template <std::size_t... Is>
        void reserve_batch(std::index_sequence<Is...>)
        {
            std::size_t reserve_n = std::min(rng_->n_, std::size_t{1000});
            (..., (std::get<Is>(*batch_).value().reserve(reserve_n)));
        }

        // move i-th element from subbatch_ to batch_
        template <std::size_t... Is>
        void move_from_subbatch(std::size_t i, std::index_sequence<Is...>)
        {
            (..., (std::get<Is>(*batch_).value().push_back(
                   std::move(std::get<Is>(*subbatch_).value()[i]))));
        }

        // find the first non-empty subbatch and return if successful
        bool find_next()
        {
            // do nothing if the end of iteration is reached
            if (subbatch_idx_ == batch_size(*subbatch_) && it_ == ranges::end(rng_->rng_)) {
                return false;
            }
            // otherwise find the first non-empty subbatch
            while (subbatch_idx_ == batch_size(*subbatch_)) {
                if (++it_ == ranges::end(rng_->rng_)) {
                    return false;
                }
                subbatch_ = std::make_shared<batch_t_>(*it_);
                subbatch_idx_ = 0;
            }
            return true;
        }

        // fill the batch_ with the elements from the current subbatch_
        void fill_batch()
        {
            reserve_batch(column_idxs_t{});
            do {
                move_from_subbatch(subbatch_idx_++, column_idxs_t{});
            } while (batch_size(*batch_) < rng_->n_ && find_next());
        }

    public:
        cursor() = default;
        explicit cursor(batch_view<Rng>& rng)
          : rng_{&rng}
          , it_{ranges::begin(rng_->rng_)}
        {
            static_assert(std::tuple_size<std::decay_t<decltype(*batch_)>>{} &&
                          "The range to be batched has to contain at least one column");
            // do nothing if the subrange is empty
            if (it_ != ranges::end(rng_->rng_)) {
                subbatch_ = std::make_shared<batch_t_>(*it_);
                // if the first subbatch is empty, try to find the next non-empty one
                if (batch_size(*subbatch_) == 0) next();
                else fill_batch();
            }
            else done_ = true;
        }

        decltype(auto) read() const
        {
            return *batch_;
        }

        bool equal(ranges::default_sentinel) const
        {
            return done_;
        }

        bool equal(const cursor& that) const
        {
            assert(rng_ == that.rng_);
            return it_ == that.it_ && subbatch_idx_ == that.subbatch_idx_;
        }

        void next()
        {
            batch_ = std::make_shared<batch_t_>();
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
/// \code
///     CXTREAM_DEFINE_COLUMN(value, int)
///     auto rng = view::iota(0, 10)
///       | create<value>(2)  // batches the data by two examples
///       | batch(3);         // changes the batch size to three examples
/// \endcode
constexpr ranges::view::view<batch_fn> batch{};

}  // namespace cxtream::stream
#endif
