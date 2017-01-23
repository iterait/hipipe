/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/tensorflow/load_graph.hpp>

#include <tensorflow/core/framework/graph.pb.h>

#include <exception>

namespace cxtream::tensorflow {

namespace fs = std::experimental::filesystem;

std::unique_ptr<::tensorflow::Session> load_graph(fs::path graph_file_name) {
  assert(fs::exists(graph_file_name));
  ::tensorflow::Status status;
  ::tensorflow::GraphDef graph_def;
  // read frozen graph
  status = ReadBinaryProto(::tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!status.ok()) {
    auto msg = std::string{"Failed to load graph at '"} +
               std::string{graph_file_name} + "': " +
               status.ToString();
    throw std::runtime_error(msg);
  }
  // create session out of the graph
  std::unique_ptr<::tensorflow::Session> session{
      ::tensorflow::NewSession(::tensorflow::SessionOptions())};
  status = session->Create(graph_def);
  if (!status.ok()) {
    auto msg = std::string{"Failed to create session from graph at '"} +
               std::string{graph_file_name} + "': " +
               status.ToString();
    throw std::runtime_error(msg);
  }
  return session;
}

}  // namespace cxtream::tensorflow
