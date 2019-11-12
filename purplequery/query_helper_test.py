# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from typing import List, Tuple  # noqa: F401

from ddt import data, ddt, unpack

from purplequery.bq_abstract_syntax_tree import EMPTY_NODE, AbstractSyntaxTreeNode
from purplequery.bq_types import BQScalarType
from purplequery.evaluatable_node import Value
from purplequery.query_helper import (AppliedRuleOutputType, RuleType, apply_rule,  # noqa: F401
                                      separated_sequence)
from purplequery.terminals import identifier, literal


class TestNode(AbstractSyntaxTreeNode):
    '''A class used only for testing apply_rule on an AbstractSyntaxTreeNode'''

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if isinstance(other, TestNode):
            return (self.a == other.a) and (self.b == other.b)
        return False


@ddt
class QueryHelperTest(unittest.TestCase):

    @data(dict(rule='SELECT',
               tokens=['SELECT', '*', 'FROM', 'TABLE'],
               result=('SELECT', ['*', 'FROM', 'TABLE']),
               comment='Rule defined by string'),

          dict(rule=('ORDER', 'BY'),
               tokens=['ORDER', 'BY', 'SomeField'],
               result=((), ['SomeField']),
               comment='Rule defined by tuple'),

          dict(rule=['FROM', 'WHERE'],
               tokens=['FROM', 'SomeTable'],
               result=('FROM', ['SomeTable']),
               comment='Rule defined by list'),

          dict(rule=['FROM', 'WHERE', None],
               tokens=['SomethingElse'],
               result=(EMPTY_NODE, ['SomethingElse']),
               comment='Rule defined by optional list'),

          dict(rule=None,
               tokens=['a', 'b', 'c'],
               result=(EMPTY_NODE, ['a', 'b', 'c']),
               comment='Rule is None'),

          dict(rule=literal,
               tokens=['1.23'],
               result=(Value(1.23, BQScalarType.FLOAT), []),
               comment='Rule defined by a method'),

          dict(rule=(TestNode, identifier, identifier),
               tokens=['a', 'b'],
               result=(TestNode('a', 'b'), []),
               comment='Rule defined by an abstract syntax tree node'),

          dict(rule='SELECT',
               tokens=['WHERE'],
               result=(None, ['WHERE']),
               comment='Rule defined by string does not match'),

          dict(rule=('ORDER', 'BY'),
               tokens=['ORDER', 'WITH', 'SomeField'],
               result=(None, ['ORDER', 'WITH', 'SomeField']),
               comment='Rule defined by tuple does not match'),

          dict(rule=['FROM', 'WHERE'],
               tokens=['SomeTable'],
               result=(None, ['SomeTable']),
               comment='Rule defined by list does not match'))
    @unpack
    def test_apply_rule(self, rule,  # type: RuleType
                        tokens,  # type: List[str]
                        result,  # type: AppliedRuleOutputType
                        comment  # type: str
                        ):
        # test: (...) -> None
        self.assertEqual(apply_rule(rule, tokens), result)

    @data(dict(rule=identifier,
               separator=['ASC', 'DESC'],
               tokens=['a', 'ASC', 'b', 'DESC'],
               result=(('a', 'b'), []),
               comment='Rule is a method, separator is either ASC or DESC'),

          dict(rule=literal,
               separator=',',
               tokens=['1', ',', '2', ',', '3'],
               result=((Value(1, BQScalarType.INTEGER),
                        Value(2, BQScalarType.INTEGER),
                        Value(3, BQScalarType.INTEGER)), []),
               comment='Rule is a method, separator is a comma'),

          dict(rule=(identifier, '=', identifier),
               separator='AND',
               tokens=['field1', '=', 'field2', 'AND', 'field3', '=', 'field4'],
               result=((('field1', 'field2'), ('field3', 'field4')), []),
               comment='Rule is a more complex tuple, separator is AND'))
    @unpack
    def test_separated_sequence(self, rule,  # type: RuleType
                                separator,  # type: RuleType
                                tokens,  # type: List[str]
                                result,  # type: AppliedRuleOutputType
                                comment  # type: str
                                ):
        # type: (...) -> None
        self.assertEqual(separated_sequence(rule, separator)(tokens), result)

    def test_separated_sequence_keep_separator(self):
        '''Test keep_separator parameter'''
        sequence_check = separated_sequence(identifier, ',', keep_separator=True)
        self.assertEqual(sequence_check(['a', ',', 'b']),
                         (('a', ',', 'b'), []))

    def test_separated_sequence_wrapper(self):
        '''Test wrapper parameter'''
        sequence_check = separated_sequence(identifier, ',', wrapper=list)
        self.assertEqual(sequence_check(['a', ',', 'b']),
                         (['a', 'b'], []))


if __name__ == '__main__':
    unittest.main()
