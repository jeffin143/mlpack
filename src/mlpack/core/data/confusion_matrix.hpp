/**
 * @file confusion_matrix_impl.hpp
 * @author Jeffin Sam
 *
 * Compute confusion matrix to evaluate the accuracy of a classification.
 * The function works only for discrete data/categorical data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP
#define MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * A confusion matrix is a summary of prediction results on a classification
 * problem.  The number of correct and incorrect predictions are summarized
 * by count and broken down by each class.
 * For example, for 2 classes, the function call will be
 *
 * @code
 * ConfusionMatrix(predictors, responses, output, 2)
 * @endcode
 *
 * In this case, the output matrix will be of size 2 * 2:
 *
 * @code
 *         0     1
 *    0    TP    FN
 *    1    FP    TN
 * @endcode
 *
 * The confusion matrix for two labels will look like what is shown above.  In
 * this confusion matrix, TP represents the number of true positives, FP
 * represents the number of false positives, FN represents the number of false
 * negatives, and TN represents the number of true negatives.
 *
 * When generalizing to 2 or more classes, the row index of the confusion matrix
 * represents the predicted classes and column index represents the actual
 * class.
 *
 * @param predictors Vector of data points.
 * @param responses The measured data for each point.
 * @param output Matrix which is represented as confusion matrix.
 * @param numClasses Number of classes.
 */
template<typename eT>
void ConfusionMatrix(const arma::Row<size_t> predictors,
                     const arma::Row<size_t> responses,
                     arma::Mat<eT>& output,
                     const size_t numClasses);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "confusion_matrix_impl.hpp"

#endif
