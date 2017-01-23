/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_TENSORFLOW_LOAD_GRAPH_HPP
#define CXTREAM_TENSORFLOW_LOAD_GRAPH_HPP

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

#include <experimental/filesystem>
#include <memory>

namespace cxtream::tensorflow {

/// \ingroup Tensorflow
/// \brief Make a session out of tensorflow frozen graph.
///
/// \param graph_file_name File with the frozen graph.
/// \returns A pointer to a tensorflow session with the graph.
std::unique_ptr<::tensorflow::Session>
load_graph(std::experimental::filesystem::path graph_file_name);

}  // namespace cxtream::tensorflow
#endif
