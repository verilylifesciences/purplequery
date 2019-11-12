#!/bin/bash

# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


set -o nounset
set -o errexit
set -o xtrace

for version in 2 3; do

  virtualenv --system-site-packages -p python$version virtualTestEnv
  # Work around virtual env error 'PS1: unbound variable'
  set +o nounset
  source virtualTestEnv/bin/activate
  set -o nounset

  export TEST_PROJECT=""

  pip$version install --upgrade pip
  pip$version install --upgrade setuptools
  pip$version install --upgrade enum34
  # analysis-py-utils necessary for bq_client test, which tests integration
  # between purplequery and analysis-py-utils
  pip$version install "git+https://github.com/verilylifesciences/analysis-py-utils@11b06535c4d3973670d5b23ae3846f3601f00a1a"
  pip$version install .

  python$version -m purplequery.bq_abstract_syntax_tree_test
  python$version -m purplequery.bq_client_test
  python$version -m purplequery.bq_operator_test
  python$version -m purplequery.bq_types_test
  python$version -m purplequery.client_test
  python$version -m purplequery.dataframe_node_test
  python$version -m purplequery.evaluatable_node_test
  python$version -m purplequery.grammar_test
  python$version -m purplequery.join_test
  python$version -m purplequery.query_helper_test
  python$version -m purplequery.query_test
  python$version -m purplequery.terminals_test
  python$version -m purplequery.tokenizer_test
  python$version -m purplequery.type_grammar_test

  deactivate
done
