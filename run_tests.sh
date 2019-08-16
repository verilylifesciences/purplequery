#!/bin/bash

# Copyright 2019 Verily Life Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
