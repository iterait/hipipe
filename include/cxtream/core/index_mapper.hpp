/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup IndexMapper Bidirectional map between indices and values.

#ifndef CXTREAM_CORE_INDEX_MAPPER_HPP
#define CXTREAM_CORE_INDEX_MAPPER_HPP

#include <range/v3/view/transform.hpp>

#include <unordered_map>
#include <vector>

namespace cxtream {

/// \ingroup IndexMapper
/// \brief Provides a bidirectional access from values to their indices in an std::vector.
template<typename T>
class index_mapper {
public:
    index_mapper() = default;

    /// Construct index mapper from a range of values.
    ///
    /// Behaves as if the values were inserted to the mapper using insert().
    template<typename Rng>
    index_mapper(Rng&& values)
    {
        insert(std::forward<Rng>(values));
    }

    /// Behaves as if the values were inserted to the mapper using insert().
    index_mapper(std::initializer_list<T> values)
    {
        insert(ranges::view::all(values));
    }

    // index_for //

    /// Returns the index of the given value.
    /// \throws std::out_of_range If the value does not exist.
    std::size_t index_for(const T& val) const
    {
        if (!contains(val)) {
            throw std::out_of_range{"The index_mapper does not contain the given value."};
        }
        return val2idx_.at(val);
    }

    /// Returns the index of the given value or a default value if it does not exist.
    std::size_t index_for(const T& val, std::size_t defval) const
    {
        auto pos = val2idx_.find(val);
        if (pos == val2idx_.end()) return defval;
        return pos->second;
    }

    /// Returns the indexes of the given values.
    /// \throws std::out_of_range If any of the values does not exist.
    std::vector<std::size_t> index_for(const std::vector<T>& vals) const
    {
        return vals
          | ranges::view::transform([this](const T& val) {
                return this->index_for(val);
            });
    }

    /// Returns the indexes of the given values or a default value if they do not exist.
    std::vector<std::size_t> index_for(const std::vector<T>& vals, std::size_t defval) const
    {
        return vals
          | ranges::view::transform([this, defval](const T& val) {
                return this->index_for(val, defval);
            });
    }

    // at //

    /// Returns the value at the given index.
    /// \throws std::out_of_range If the index does not exist in the mapper.
    const T& at(const std::size_t& idx) const
    {
        if (idx >= size()) {
            throw std::out_of_range{"Index " + std::to_string(idx) + " cannot be found in "
                                    "index_mapper of size " + std::to_string(size()) + "."};
        }
        return idx2val_[idx];
    }


    /// Returns the values at the given indexes.
    /// \throws std::out_of_range If any of the indexes does not exist in the mapper.
    std::vector<T> at(const std::vector<std::size_t>& idxs) const
    {
        return idxs
          | ranges::view::transform([this](std::size_t idx) {
                return this->at(idx);
            });
    }

    // insert //

    /// Inserts a value into the mapper with index size()-1.
    /// \param val The value to be inserted.
    /// \returns The index of the inserted value.
    /// \throws std::invalid_argument if the element is already present in the mapper.
    std::size_t insert(T val)
    {
        if (contains(val)) {
            throw std::invalid_argument{"The element is already present in the index mapper."};
        }
        val2idx_[val] = idx2val_.size();
        idx2val_.push_back(std::move(val));
        return idx2val_.size() - 1;
    }

    /// Insert elements from a range to the index mapper.
    ///
    /// If this function throws an exception, the state of the mapper is undefined.
    ///
    /// \param rng The range of values to be inserted.
    /// \throws std::invalid_argument if any of the elements is already present in the mapper.
    template<typename Rng, CONCEPT_REQUIRES_(ranges::Container<Rng>())>
    void insert(Rng rng)
    {
        for (auto& val : rng) insert(std::move(val));
    }

    /// Specialization of range insertion for views.
    template<typename Rng, CONCEPT_REQUIRES_(ranges::View<Rng>())>
    void insert(Rng rng)
    {
        for (auto&& val : rng) insert(std::forward<decltype(val)>(val));
    }

    // try_insert //

    /// Tries to insert a value into the mapper with index size()-1. If the value is
    /// already present in the mapper, the operation does nothing.
    ///
    /// \param val The value to be inserted.
    /// \returns True if the value insertion was successful.
    bool try_insert(T val)
    {
        if (contains(val)) return false;
        insert(std::move(val));
        return true;
    }

    /// Insert elements to index mapper while skipping duplicates.
    ///
    /// \param rng The range of values to be inserted.
    template<typename Rng, CONCEPT_REQUIRES_(ranges::Container<Rng>())>
    void try_insert(Rng rng)
    {
        for (auto& val : rng) try_insert(std::move(val));
    }

    /// Specialization of try_insert for views.
    template<typename Rng, CONCEPT_REQUIRES_(ranges::View<Rng>())>
    void try_insert(Rng rng)
    {
        for (auto&& val : rng) try_insert(std::forward<decltype(val)>(val));
    }

    // helper functions //

    /// Checks whether the mapper contains the given value.
    bool contains(const T& val) const
    {
        return val2idx_.count(val);
    }

    /// Returns the size of the mapper.
    std::size_t size() const
    {
        return val2idx_.size();
    }

    // data access //

    /// Returns all the contained values.
    const std::vector<T>& values() const
    {
        return idx2val_;
    }

private:
    std::unordered_map<T, std::size_t> val2idx_;
    std::vector<T> idx2val_;
};

/// \ingroup IndexMapper
/// \brief Make index mapper from unique elements of a range.
///
/// Example:
/// \code
///     std::vector<std::string> data = {"bum", "bada", "bum", "bum", "bada", "yeah!"};
///     index_mapper<std::string> mapper = make_unique_index_mapper(data);
///     // mapper.values() == {"bum", "bada", "yeah!"}
/// \endcode
///
/// \param rng The range of values to be inserted.
/// \returns The index mapper made of unique values of the range.
template<typename Rng, typename T = ranges::range_value_type_t<Rng>>
index_mapper<T> make_unique_index_mapper(Rng&& rng)
{
    index_mapper<T> mapper;
    mapper.try_insert(std::forward<Rng>(rng));
    return mapper;
}

}  // end namespace cxtream
#endif
