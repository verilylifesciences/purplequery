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

import re
import unittest
from typing import List, Union  # noqa: F401

from ddt import data, ddt, unpack

from bq_abstract_syntax_tree import EvaluationContext
from bq_operator import (BINARY_OPERATOR_PATTERN, _reparse_binary_expression,
                         binary_operator_expression_rule)
from bq_types import BQScalarType
from evaluatable_node import Value  # noqa: F401
from terminals import literal


@ddt
class BqOperatorTest(unittest.TestCase):

    @data(
        ('+-', ['+', '-']),
        ('<-', ['<', '-']),
        ('>-', ['>', '-']),
        ('<<', ['<<']),
        ('<=', ['<=']),
        ('<>', ['<>']),
        ('>=', ['>=']),
        ('!=', ['!=']),
    )
    @unpack
    def test_tokenize_binary_operators(self, s, tokens):
        # type: (str, List[str]) -> None
        """Test that two-character operators are lexed as one token, but two operators aren't."""
        self.assertEqual(re.findall(BINARY_OPERATOR_PATTERN, s), tokens)

    @data(
        ('1 + 2 * 3', '(+ 1 (* 2 3))', 7),
        ('1 * 2 + 3', '(+ (* 1 2) 3)', 5),
        ('1 + 2 * 3 - 4', '(- (+ 1 (* 2 3)) 4)', 3),
        ('1 + 2 * 3 << 4', '(<< (+ 1 (* 2 3)) 4)', (7 << 4)),
        ('1 + 2 << 3 * 4', '(<< (+ 1 2) (* 3 4))', (3 << 12)),
        ('1 * 2 + 3 << 4', '(<< (+ (* 1 2) 3) 4)', (5 << 4)),
        ('1 * 2 << 3 + 4', '(<< (* 1 2) (+ 3 4))', (2 << 7)),
        ('1 << 2 * 3 + 4', '(<< 1 (+ (* 2 3) 4))', (1 << 10)),
        ('1 << 2 + 3 * 4', '(<< 1 (+ 2 (* 3 4)))', (1 << 14)),
    )
    @unpack
    def test_binary_expressions(self, expression_str, strexpr, result):
        # type: (str, str, int) -> None
        tokens = re.findall('|'.join((BINARY_OPERATOR_PATTERN, '\d+')), expression_str)
        node, leftover = binary_operator_expression_rule(literal)(tokens)
        self.assertFalse(leftover)
        self.assertEqual(node.strexpr(), strexpr)
        self.assertEqual(list(node.evaluate(context=EvaluationContext({})).series)[0], result)

    @data(
        ('12 / 3', 4),
        ('3 + 4.0', 7.0),
        ('3 << 4', 48),
        ('12 >> 2', 3),
        ('7 & 3', 3),
        ('5 | 3', 7),
        ('7 ^ 3', 4),
        ('3 = 3', True),
        ('3 = 4', False),
        ('3 != 3', False),
        ('3 != 4', True),
        ('3 <> 3', False),
        ('3 <> 4', True),
        ('3 < 3', False),
        ('3 < 4', True),
        ('3 <= 3', True),
        ('3 <= 2', False),
        ('3 > 3', False),
        ('4 > 3', True),
        ('3 >= 3', True),
        ('2 >= 3', False),
        ('3 = 3 AND 4 = 4', True),
        ('3 = 3 AND 3 = 4', False),
        ('3 = 2 AND 3 = 4', False),
        ('3 = 3 OR 4 = 4', True),
        ('3 = 3 OR 3 = 4', True),
        ('3 = 2 OR 3 = 4', False),
    )
    @unpack
    def test_binary_operators(self, expression_str, result):
        # type: (str, Union[int, bool]) -> None
        tokens = re.findall('|'.join((BINARY_OPERATOR_PATTERN, '\d+\.\d+', '\d+', 'AND', 'OR')),
                            expression_str)
        node, leftover = binary_operator_expression_rule(literal)(tokens)
        self.assertFalse(leftover)
        self.assertEqual(list(node.evaluate(context=EvaluationContext({})).series)[0], result)

    def test_even_length_sequence_raises(self):
        with self.assertRaisesRegexp(ValueError, 'Sequence must be of odd length'):
            _reparse_binary_expression([Value(3, BQScalarType.INTEGER), '+'])

    def test_all_string_sequence_raises(self):
        with self.assertRaisesRegexp(ValueError, 'Sequence must alternate node, operator, node'):
            _reparse_binary_expression(['+', '*', '-'])

    def test_not_an_operator(self):
        with self.assertRaisesRegexp(ValueError, 'Unknown operator string bar'):
            _reparse_binary_expression([Value(3, BQScalarType.INTEGER),
                                        'bar',
                                        Value(4, BQScalarType.INTEGER),
                                        ])


if __name__ == '__main__':
    unittest.main()
