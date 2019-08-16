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

'''Grammar rules representing the terminals in Google BigQuery Standard SQL syntax.

This module exports functions usable as grammar rules in a recursive descent parser for the
terminals of the language -- the rules that do not depend on other rules, but directly produce
tokens.

There are terminals of three types:
  - literals (numbers, strings, constants like TRUE and NULL)
  - identifiers
  - reserved words
'''

import re
from typing import Callable, List, Optional, Tuple  # noqa: F401

from bq_types import BQScalarType
from evaluatable_node import Value
from patterns import (BACKTICK_PATTERN, FLOAT_LITERAL_PATTERNS, IDENTIFIER_PATTERN,
                      INT_LITERAL_PATTERN, STR_LITERAL_PATTERNS)

# Reserved words that are used in the grammar, not available as identifiers.
# This list is from
# https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#reserved-keywords
RESERVED_WORDS = frozenset(('ALL AND ANY ARRAY AS ASC ASSERT_ROWS_MODIFIED AT BETWEEN BY CASE CAST '
                            'COLLATE CONTAINS CREATE CROSS CUBE CURRENT DEFAULT DEFINE DESC '
                            'DISTINCT ELSE END ENUM ESCAPE EXCEPT EXCLUDE EXISTS EXTRACT FALSE '
                            'FETCH FOLLOWING FOR FROM FULL GROUP GROUPING GROUPS HASH HAVING IF '
                            'IGNORE IN INNER INTERSECT INTERVAL INTO IS JOIN LATERAL LEFT LIKE '
                            'LIMIT LOOKUP MERGE NATURAL NEW NO NOT NULL NULLS OF ON OR ORDER OUTER '
                            'OVER PARTITION PRECEDING PROTO RANGE RECURSIVE RESPECT RIGHT ROLLUP '
                            'ROWS SELECT SET SOME STRUCT TABLESAMPLE THEN TO TREAT TRUE UNBOUNDED '
                            'UNION UNNEST USING WHEN WHERE WINDOW WITH WITHIN').split())


def identifier(tokens):
    # type: (List[str]) -> Tuple[Optional[str], List[str]]
    """Checks if the first token is an identifier.

    Args:
        tokens: List of tokens, to decide whether tokens[0] is an identifier.
    Returns:
        Tuple of identifier (if there was a match) and rest of unidentified tokens.
    """
    if tokens:
        maybe_id = tokens[0]
        if (re.search(r'^' + IDENTIFIER_PATTERN + '$', maybe_id)
                and maybe_id.upper() not in RESERVED_WORDS):
            return maybe_id, tokens[1:]
        # If enclosed in backticks, identifiers can contain any character,
        # including spaces, and can even be reserved words.
        if re.search(r'^' + BACKTICK_PATTERN + '$', maybe_id):
            # Remove backticks before returning
            return maybe_id[1:-1], tokens[1:]
    return None, tokens


_CONSTANTS = {
    'NULL': (None, None),
    'TRUE': (True, BQScalarType.BOOLEAN),
    'FALSE': (False, BQScalarType.BOOLEAN),
}


def literal(tokens):
    # type: (List[str]) -> Tuple[Optional[Value], List[str]]
    """Checks if the first token is a literal (number, string, boolean, null).

    Args:
        tokens: List of tokens, to decide whether tokens[0] is a literal.
    Returns:
        Tuple of parsed literal (if there was a match) and rest of unidentified tokens.
    """
    if tokens:
        token = tokens[0]
        rest_of_tokens = tokens[1:]
        float_pattern = '|'.join([r'^' + pattern + '$' for pattern in FLOAT_LITERAL_PATTERNS])
        if re.search(r'^' + float_pattern + '$', token):
            return Value(float(token), BQScalarType.FLOAT), rest_of_tokens
        if re.search(r'^' + INT_LITERAL_PATTERN + '$', token):
            return Value(int(tokens[0]), BQScalarType.INTEGER), rest_of_tokens
        str_pattern = '|'.join([r'^' + pattern + '$' for pattern in STR_LITERAL_PATTERNS])
        if re.search(r'^' + str_pattern + '$', token):
            # Remove quotes before returning
            return Value(token[1:-1], BQScalarType.STRING), rest_of_tokens
        token = token.upper()
        if token in _CONSTANTS:
            return Value(*_CONSTANTS[token]), rest_of_tokens
    return None, tokens


def grammar_literal(*words):
    # type: (*str) -> Callable[[List[str]], Tuple[Optional[str], List[str]]]
    """Checks if the first token(s) is a grammar literal. This includes all the reserved words
    above, as well as binary operators, parentheses, brackets, etc.

    Args:
        words: Grammatic literal(s) to match.
    Returns:
        Function that checks whether given tokens match given grammatic literals.
    """

    def match_strings_to_tokens(tokens):
        # type: (List[str]) -> Tuple[Optional[str], List[str]]
        """Checks if the next token(s) match the previously given grammatic literal(s).

        Args:
            tokens: List of tokens, to decide whether the first token matches a grammatic literal.
        Returns:
            Tuple of grammatic literal(s) (if there was a match, joined by an underscore if there
            is more than one), and the rest of the unidentified tokens.
        """

        # Convert all requested words to upper case
        words_upper = tuple(word.upper() for word in words)

        # If there are enough tokens for the given reserved word(s) AND
        # the first n tokens match the given reserved word(s)
        if (len(tokens) >= len(words) and
                words_upper == tuple(token.upper() for token in tokens[:len(words)])):

            # Put underscores between the found reserved words, and return the
            # unused tokens
            return '_'.join(words_upper), tokens[len(words):]
        return None, tokens

    return match_strings_to_tokens
