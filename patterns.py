# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Patterns to tokenize a Google BigQuery Standard SQL query.

Spec is here: https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical
"""
IDENTIFIER_PATTERN = '[A-Za-z_][A-Za-z0-9_]*'
BACKTICK_PATTERN = '`[^`]+`'
STR_LITERAL_PATTERNS = ['"[^"]*"', "'[^']*'"]
INT_LITERAL_PATTERN = r'\d+'
FLOAT_LITERAL_PATTERNS = [
    # Digits, decimal point, optional digits, optional exponent
    r'\d+\.\d*(?:[eE][+-]?\d+)?',
    # Optional digits, decimal point, digits, optional exponent
    r'\d*\.\d+(?:[eE][+-]?\d+)?',
    # Digits, exponent
    r'\d+[eE][+-]?\d+']
COMMENT_PATTERN = '--.*$'
NON_OPERATOR_TOKEN_PATTERN = r'[().,!]|\[|\]'
