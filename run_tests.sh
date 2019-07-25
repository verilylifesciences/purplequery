#!/bin/bash

# TODO: License

set -o nounset
set -o errexit
set -o xtrace

for version in 2 3; do

  virtualenv --system-site-packages -p python$version virtualTestEnv
  # Work around virtual env error 'PS1: unbound variable'
  set +o nounset
  source virtualTestEnv/bin/activate
  set -o nounset

  pip$version install --upgrade pip
  pip$version install --upgrade setuptools
  pip$version install --upgrade enum34
  pip$version install .

  python$version -m bq_abstract_syntax_tree_test
  python$version -m bq_client_test
  python$version -m bq_operator_test
  python$version -m bq_types_test
  python$version -m client_test
  python$version -m dataframe_node_test
  python$version -m evaluatable_node_test
  python$version -m grammar_test
  python$version -m join_test
  python$version -m query_helper_test
  python$version -m query_test
  python$version -m terminals_test
  python$version -m tokenizer_test

  deactivate
done
