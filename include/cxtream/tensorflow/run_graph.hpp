/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_TENSORFLOW_RUN_GRAPH_HPP
#define CXTREAM_TENSORFLOW_RUN_GRAPH_HPP

#include <cxtream/core/utility/tuple.hpp>
#include <cxtream/tensorflow/utility/to_tf_type.hpp>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

#include <exception>
#include <experimental/filesystem>
#include <string>
#include <vector>

namespace cxtream::tensorflow {

/// \ingroup Tensorflow
/// \brief Run a session and feed its graph with the provided inputs.
///
/// The session can be obtained using load_graph().
///
/// The lengths of input_names, input_shapes and input_data arguments have to be equal.
/// The lengths of output_names and the OutT template parameter list have to be equal.
///
/// \param session Pointer to the session to be run.
/// \param input_names Names of the input variables for the TF graph.
/// \param input_data The data to be fed to the selected input variables.
/// \param input_shapes Shapes of the input variables including the batch dimension.
/// \param output_names The names of the variables to be extracted from the TF graph.
/// \returns Tuple of output data and their corresponding shapes.
template <typename... OutTs, typename... InTs>
std::tuple<std::tuple<std::vector<OutTs>...>, std::vector<std::vector<long>>>
run_graph(::tensorflow::Session& session,
          const std::vector<std::string>& input_names,
          const std::tuple<std::vector<InTs>...>& input_data,
          const std::vector<std::vector<long>>& input_shapes,
          const std::vector<std::string>& output_names)
{
  assert(ranges::size(input_names) == ranges::size(input_shapes));
  assert(ranges::size(input_names) == sizeof...(InTs));
  assert(ranges::size(output_names) == sizeof...(OutTs));
  std::tuple<InTs...> in_types;
  std::tuple<OutTs...> out_types;

  // convert the input data to tensors
  std::vector<std::pair<std::string, ::tensorflow::Tensor>> feed;
  utility::tuple_for_each_with_index(input_data, [&](auto& data, auto i) {
      // build shape
      ::tensorflow::TensorShape shape;
      for (long val : input_shapes[i]) shape.AddDim(val);
      // build data type
      auto dtype = to_tf_type(std::tuple_element_t<i, decltype(in_types)>{});
      // allocate tensor
      ::tensorflow::Tensor tensor{dtype, shape};
      // copy data into tensor
      std::copy(data.begin(), data.end(),
                tensor.flat<std::tuple_element_t<i, decltype(in_types)>>().data());
      // link the tensor with the corresponding name
      feed.emplace_back(input_names[i], std::move(tensor));
  });

  // run the graph
  std::vector<::tensorflow::Tensor> outputs;
  ::tensorflow::Status status = session.Run(feed, output_names, {}, &outputs);
  if (!status.ok()) {
    auto msg = std::string{"Failed to run tensorflow graph: "} + status.ToString();
    throw std::runtime_error(msg);
  }

  // convert the result to std::vectors
  std::tuple<std::vector<OutTs>...> raw_outputs;
  std::vector<std::vector<long>> output_shapes;
  utility::tuple_for_each_with_index(raw_outputs, [&](auto& raw_output, auto i) {
      ::tensorflow::Tensor& output = outputs[i];
      // allocate space in std::vector
      raw_output.resize(output.NumElements());
      // copy the data from tensor to std::vector
      std::copy_n(output.flat<std::tuple_element_t<i, decltype(out_types)>>().data(),
                  output.NumElements(),
                  raw_output.begin());
      // convert the shape of the tensor to std::vector
      std::vector<long> output_shape;
      for (long d = 0; d < output.dims(); ++d) output_shape.push_back(output.dim_size(d));
      output_shapes.emplace_back(std::move(output_shape));
      // for (long s : output.shape()) output_shapes.back().push_back(s);
      // (std::vector<long>(output.shape().begin(), output.shape().end()));
  });

  return {std::move(raw_outputs), std::move(output_shapes)};
}

}  // namespace cxtream::tensorflow
#endif
