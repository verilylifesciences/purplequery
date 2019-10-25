# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Information about binary operators.

Includes the operator's: string representation, precedence, function, and
result type (if known in advance).
"""

import operator
from typing import Callable, NamedTuple, Optional

import numpy as np

from bq_types import BQScalarType, BQType

"""Tuple for storing operator info.

    Attributes:
        operator: The character(s) that represent this operator.
        precedence: Lower number means earlier in order of operations.
        function: The actual function describing the action the operator performs.
        result_type: The type of the operation's result, or None if type depends on
            additional information.
"""
_OperatorInfo = NamedTuple('_OperatorInfo', [
    ('operator', str),
    ('precedence', int),
    ('function', Callable),
    ('result_type', Optional[BQType])])

# Information on binary operators in BigQuery
# Precedence is from https://cloud.google.com/bigquery/docs/reference/standard-sql/operators
# A lower number binds more tightly.
BINARY_OPERATOR_INFO = {info.operator: info for info in (
    _OperatorInfo('*', 3, operator.mul, None),
    _OperatorInfo('/', 3, operator.truediv, None),
    _OperatorInfo('+', 4, operator.add, None),
    _OperatorInfo('-', 4, operator.sub, None),
    _OperatorInfo('<<', 5, np.left_shift, None),
    _OperatorInfo('>>', 5, np.right_shift, None),
    _OperatorInfo('&', 6, operator.and_, None),
    _OperatorInfo('^', 7, operator.xor, None),
    _OperatorInfo('|', 8, operator.or_, None),
    _OperatorInfo('=', 9, operator.eq, BQScalarType.BOOLEAN),
    _OperatorInfo('<', 9, operator.lt, BQScalarType.BOOLEAN),
    _OperatorInfo('>', 9, operator.gt, BQScalarType.BOOLEAN),
    _OperatorInfo('<=', 9, operator.le, BQScalarType.BOOLEAN),
    _OperatorInfo('>=', 9, operator.ge, BQScalarType.BOOLEAN),
    _OperatorInfo('!=', 9, operator.ne, BQScalarType.BOOLEAN),
    _OperatorInfo('<>', 9, operator.ne, BQScalarType.BOOLEAN),
    # Not included here, will be specified in separate grammar:
    # _OperatorInfo('LIKE', 9, lambda a, b: bool(re.search(b, a)), BQScalarType.BOOLEAN),
    # NOT like, [NOT] BETWEEN, [NOT] IN, and IS [NOT]
    # Also _OperatorInfo('NOT', 10) (and other unary operators)
    _OperatorInfo('AND', 11, operator.and_, BQScalarType.BOOLEAN),
    _OperatorInfo('OR', 12, operator.or_, BQScalarType.BOOLEAN),
)}
