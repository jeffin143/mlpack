/**
 * @file word_2_vec.hpp
 * @author Jeffin Sam
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_WORD_2_VECTOR_HPP
#define MLPACK_CORE_DATA_WORD_2_VECTOR_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"

namespace mlpack {
namespace data {

class word2vec
{
 public:

  word2vec(double learningRate = 0.1, size_t windowSize = 2, size_t epochs = 50,
           int batchSize = 50, int embeddingSize = 100, int iterationPerCycle = 100000,
           double stepSize = 1e-4, bool cbow = false)
      :learningRate(learningRate),
       windowSize(windowSize),
       epochs(epochs),
       batchSize(batchSize),
       embeddingSize(embeddingSize),
       iterationPerCycle(iterationPerCycle),
       stepSize(stepSize),
       cbow(cbow);
    {}

  template<typename TokenizerType>
  void Fit(const std::string& corpus,const TokenizerType& tokenizer)
  {
    boost::string_view token;
    boost::string_view strView(corpus);
    token = tokenizer(strView);
    while (!token.empty())
    {
      tokenizedcorpus.push_back(string(token));
      if (mapping.find(token) == mapping.end())
      {
        tokens.push_back(std::string(token));
        mapping[tokens.back()] = tokens.size()-1;
        reverseMapping[tokens.size()-1] = tokens.back();
      }
      token = tokenizer(strView);
    }
    Create();
  }

  void Create()
  {
    x.resize(tokens.size(),tokenizedcorpus.size());
    y.resize(tokens.size(),tokenizedcorpus.size());
    for (int i = 0 ; i < tokenizedcorpus.size(); i++)
    {
      x(mapping[tokenizedcorpus[i]] , i)++;
      for (int j = 1; j <= windowSize; j++)
      {
        if(i-j>0)
          y(mapping[tokenizedcorpus[i-j]], i)++;
        if(i+j<tokenizedcorpus.size())
          y(mapping[tokenizedcorpus[ i + j]], i)++;
      }
    }
    if(cbow)
    {
      arma::mat temp = std::move(x);
      x = y;
      y = std::move(temp);
    }
  }
  void Train()
  {
    FFN<CrossEntropyError<>,RandomInitialization> model;
    model.Add<Linear<> >(x.n_rows,embeddingSize);
    model.Add<Linear<> >(embeddingSize, y.n_rows);
    // You can change this to softmax after sreenik's PR1958 is merged.
    model.Add<LogSoftMax<> >();
   // Setting parameters Stochastic Gradient Descent (SGD) optimizer.
    SGD<AdamUpdate> optimizer(
      // Step size of the optimizer.
      stepSize,
      // Batch size. Number of data points that are used in each iteration.
      batchSize,
      // Max number of iterations
      iterationPerCycle,
      // Tolerance, used as a stopping condition. This small number
      // means we never stop by this condition and continue to optimize
      // up to reaching maximum of iterations.
      1e-8,
      // Shuffle. If optimizer should take random data points from the dataset at
      // each iteration.
      true,
      // Adam update policy.
      AdamUpdate(1e-8, 0.9, 0.999));

    // Cycles for monitoring the process of a solution.
    for (int i = 1; i <= epochs; i++)
    {
      // Train neural network. If this is the first iteration, weights are
      // random, using current values as starting point otherwise.
      model.Train(x, y, optimizer);

      // Don't reset optimizer's parameters between cycles.
      optimizer.ResetPolicy() = false;
    }
}

 private:
  std::unordered_map<boost::string_view, size_t, boost::hash<boost::string_view>> mapping;
  std::unordered_map<size_t , boost::string_view> reverseMapping;
  std::deque<std::string> tokens;
  arma::mat x;
  arma::mat y;
  double learningRate ;
  size_t windowSize;
  size_t epochs;
  size_t batchSize;
  size_t embeddingSize;
  size_t iterationPerCycle;
  double stepSize;
  vector<string> tokenizedcorpus;
  bool cbow;
};

} // namespace data
} // namespace mlpack


#endif
