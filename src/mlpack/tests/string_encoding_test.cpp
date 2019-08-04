/**
 * @file string_encoding_test.cpp
 * @author Jeffin Sam
 *
 * Tests for the StringEncoding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
#include <mlpack/core/data/tokenizers/split_by_any_of.hpp>
#include <mlpack/core/data/tokenizers/char_extract.hpp>
#include <mlpack/core/data/string_encoding.hpp>
#include <mlpack/core/data/string_encoding_policies/bow_encoding_policy.hpp>
#include <boost/test/unit_test.hpp>
#include <memory>
#include "test_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(StringEncodingTest);

//! Common input for some tests.
static vector<string> stringEncodingInput = {
    "hello how are you",
    "i am good",
    "Good how are you",
};


/**
 * Test the Bag of Words encoding algorithm.
 */
BOOST_AUTO_TEST_CASE(BowEncodingTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  arma::mat output;
  BowEncoding<SplitByAnyOf::TokenType> encoder;
  SplitByAnyOf tokenizer(" ");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t> keysCount;
  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }
  arma::mat expected = {
    { 1, 1, 1, 1, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 1, 1, 1, 0 },
    { 0, 1, 1, 1, 0, 0, 0, 1 }
  };
  CheckMatrices(output, expected);
}

/**
 * Test the one pass modification of the Bag of Words encoding algorithm.
 */
BOOST_AUTO_TEST_CASE(OnePassBowEncodingTest)
{
  using DictionaryType = StringEncodingDictionary<boost::string_view>;

  vector<vector<size_t>> output;
  BowEncoding<SplitByAnyOf::TokenType> encoder(
      (BagOfWordsEncodingPolicy()));
  SplitByAnyOf tokenizer(" ");

  encoder.Encode(stringEncodingInput, output, tokenizer);

  const DictionaryType& dictionary = encoder.Dictionary();

  // Checking that everything is mapped to different numbers
  std::unordered_map<size_t, size_t> keysCount;
  for (auto& keyValue : dictionary.Mapping())
  {
    keysCount[keyValue.second]++;
    // Every token should be mapped only once
    BOOST_REQUIRE_EQUAL(keysCount[keyValue.second], 1);
  }

  vector<vector<size_t>> expected = {
    { 1, 1, 1, 1, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 1, 1, 1, 0 },
    { 0, 1, 1, 1, 0, 0, 0, 1 }
  };

  BOOST_REQUIRE(output == expected);
}


/**
* Test Bag of Words encoding for characters using lamda function.
*/
BOOST_AUTO_TEST_CASE(BowEncodingIndividualCharactersTest)
{
  vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  arma::mat output;
  BowEncoding<CharExtract::TokenType> encoder;

  // Passing a empty string to encode characters
  encoder.Encode(input, output, CharExtract());
  arma::mat target = {
    { 1, 1, 1, 0, 0 },
    { 0, 1, 1, 1, 1 },
    { 1, 1, 0, 1, 0 }
  };

  CheckMatrices(output, target);
}

/**
 * Test the one pass modification of the Bag of Words encoding algorithm
 * in case of individual character encoding.
 */
BOOST_AUTO_TEST_CASE(OnePassBowEncodingIndividualCharactersTest)
{
  std::vector<string> input = {
    "GACCA",
    "ABCABCD",
    "GAB"
  };

  vector<vector<size_t>> output;
  BowEncoding<CharExtract::TokenType> encoder;

  // Passing a empty string to encode characters
  encoder.Encode(input, output, CharExtract());

  vector<vector<size_t>> expected = {
    { 1, 1, 1, 0, 0 },
    { 0, 1, 1, 1, 1 },
    { 1, 1, 0, 1, 0 }
  };

  BOOST_REQUIRE(output == expected);
}

BOOST_AUTO_TEST_SUITE_END();

