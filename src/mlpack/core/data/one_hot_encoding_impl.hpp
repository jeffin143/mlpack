/**
 * @file one_hot_encoding_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of one hot encoding functions; categorical variables as binary
 * vectors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_ONE_HOT_ENCODING_IMPL_HPP
#define MLPACK_CORE_DATA_ONE_HOT_ENCODING_IMPL_HPP

// In case it hasn't been included yet.
#include "one_hot_encoding.hpp"


namespace mlpack {
namespace data {

/**
 * Given a set of labels of a particular datatype, convert them to binary
 * vector. The categorical values be mapped to integer values.
 * Then, each integer value is represented as a binary vector that is
 * all zero values except the index of the integer, which is marked
 * with a 1.
 *
 * @param labelsIn Input labels of arbitrary datatype.
 * @param output Binary matrix.
 */
template<typename OutputType, typename RowType>
void OneHotEncoding(const RowType& labelsIn,
                    OutputType& output)
{
  arma::Row<size_t> labels;
  labels.set_size(labelsIn.n_elem);

  // Loop over the input labels, and develop the mapping.
  std::unordered_map<typename OutputType::elem_type, size_t> labelMap; // Map for labelsIn to labels.
  size_t curLabel = 0;
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    // If labelsIn[i] is already in the map, use the existing label.
    if (labelMap.count(labelsIn[i]) != 0)
    {
      labels[i] = labelMap[labelsIn[i]] - 1;
    }
    else
    {
      // If labelsIn[i] not there then add it to the map.
      labelMap[labelsIn[i]] = curLabel + 1;
      labels[i] = curLabel;
      ++curLabel;
    }
  }
  // Resize output matrix to necessary size, and fill it with zeros.
  output.zeros(labelsIn.n_elem, curLabel);
  // Fill ones in at the required places.
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    output(i, labels[i]) = 1;
  }
  labelMap.clear();
}

template<typename OutputType, typename InputType,
typename RowType>
void OneHotEncoding(const InputType& input,
                    const RowType& indices,
                    OutputType& output)
{
  arma::Row<size_t> labels;
  labels.set_size(input.n_cols);
  output = input;
  // Loop over the input labels, and develop the mapping.
  for (size_t row = 0 ; row = indices.n_elem; row++)
  {
    std::unordered_map<typename OutputType::elem_type, size_t> labelMap; // Map for labelsIn to labels.
    size_t curLabel = 0;
    for (size_t i = 0; i < input.n_cols; ++i)
    {
      // If labelsIn[i] is already in the map, use the existing label.
      if (labelMap.count(input(indices(row), i)) != 0)
      {
        labels[i] = labelMap[input(indices(row), i)] - 1;
      }
      else
      {
        // If labelsIn[i] not there then add it to the map.
        labelMap[input(indices(row), i)] = curLabel + 1;
        labels[i] = curLabel;
        ++curLabel;
      }
    }
    // Resize output matrix to necessary size, and fill it with zeros.
    OutputType tempOutput;
    tempOutput.zeros(curLabel, labels.n_elem);
    // Fill ones in at the required places.
    for (size_t i = 0; i < labels.n_elem; ++i)
    {
      output(labels[i], i) = 1;
    }
    labelMap.clear();
    output.shed_row(indices(row));
    output.insert_rows(indices(row), tempOutput);
  }
}

} // namespace data
} // namespace mlpack

#endif
