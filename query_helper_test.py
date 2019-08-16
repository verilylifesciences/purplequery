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

import unittest
from typing import List, Tuple  # noqa: F401

from ddt import data, ddt, unpack

import query_helper
from bq_abstract_syntax_tree import EMPTY_NODE, AbstractSyntaxTreeNode
from bq_types import BQScalarType
from evaluatable_node import Value
from terminals import identifier, literal


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
    def test_apply_rule(self, rule,  # type: query_helper.RuleType
                        tokens,  # type: List[str]
                        result,  # type: query_helper.AppliedRuleOutputType
                        comment  # type: str
                        ):
        # test: (...) -> None
        self.assertEqual(query_helper.apply_rule(rule, tokens), result)

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
    def test_separated_sequence(self, rule,  # type: query_helper.RuleType
                                separator,  # type: query_helper.RuleType
                                tokens,  # type: List[str]
                                result,  # type: query_helper.AppliedRuleOutputType
                                comment  # type: str
                                ):
        # type: (...) -> None
        self.assertEqual(query_helper.separated_sequence(rule, separator)(tokens), result)

    def test_separated_sequence_keep_separator(self):
        '''Test keep_separator parameter'''
        sequence_check = query_helper.separated_sequence(identifier, ',', keep_separator=True)
        self.assertEqual(sequence_check(['a', ',', 'b']),
                         (('a', ',', 'b'), []))

    def test_separated_sequence_wrapper(self):
        '''Test wrapper parameter'''
        sequence_check = query_helper.separated_sequence(identifier, ',', wrapper=list)
        self.assertEqual(sequence_check(['a', ',', 'b']),
                         (['a', 'b'], []))


if __name__ == '__main__':
    unittest.main()
