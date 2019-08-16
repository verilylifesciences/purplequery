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

"""Tokenize a Google BigQuery Standard SQL query."""

import re
from typing import List  # noqa: F401

from bq_operator import BINARY_OPERATOR_PATTERN
from patterns import (BACKTICK_PATTERN, COMMENT_PATTERN, FLOAT_LITERAL_PATTERNS, IDENTIFIER_PATTERN,
                      INT_LITERAL_PATTERN, NON_OPERATOR_TOKEN_PATTERN, STR_LITERAL_PATTERNS)

_COMBINED_PATTERN = re.compile(
    '|'.join((
        COMMENT_PATTERN,
        BACKTICK_PATTERN,
        '|'.join(STR_LITERAL_PATTERNS),
        BINARY_OPERATOR_PATTERN,
        '|'.join(FLOAT_LITERAL_PATTERNS),
        INT_LITERAL_PATTERN,
        NON_OPERATOR_TOKEN_PATTERN,
        IDENTIFIER_PATTERN)),
    flags=re.MULTILINE)


def tokenize(query):
    # type: (str) -> List[str]
    matches = _COMBINED_PATTERN.findall(query)
    return [match for match in matches if not match.startswith('--')]  # strip comments
