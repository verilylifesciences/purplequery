# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Tokenize a Google BigQuery Standard SQL query."""

import re
from typing import List  # noqa: F401

from .bq_operator import BINARY_OPERATOR_PATTERN
from .patterns import (BACKTICK_PATTERN, COMMENT_PATTERN, FLOAT_LITERAL_PATTERNS,
                       IDENTIFIER_PATTERN, INT_LITERAL_PATTERN, NON_OPERATOR_TOKEN_PATTERN,
                       STR_LITERAL_PATTERNS)

_COMPILED_COMMENT_PATTERN = re.compile(COMMENT_PATTERN, flags=re.MULTILINE)


def remove_comments(query):
    # type: (str) -> str
    return _COMPILED_COMMENT_PATTERN.sub('', query)


_COMBINED_PATTERN = re.compile(
    '|'.join((
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
    return _COMBINED_PATTERN.findall(remove_comments(query))
