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
from typing import List, Tuple, Union  # noqa: F401

from ddt import data, ddt, unpack

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import EMPTY_NODE, EvaluatableNode, Field, _EmptyNode  # noqa: F401
from bq_types import BQScalarType
from dataframe_node import Select
from evaluatable_node import (Case, Cast, Count, Exists, Extract, If, InCheck, Not, NullCheck,
                              UnaryNegation, Value, _AggregatingFunctionCall)
from grammar import core_expression, data_source, post_expression, query_expression, select


@ddt
class GrammarTest(unittest.TestCase):

    def test_query_expression(self):
        # type: () -> None
        ast, leftover = query_expression(
            ['SELECT', '*', 'FROM', '`my_project.my_dataset.my_table`'])
        self.assertEqual(leftover, [])

        self.assertEqual(ast.base_query.__class__, Select)

    def test_select(self):
        # type: () -> None
        ast, leftover = select(['SELECT', '*', 'FROM', '`my_project.my_dataset.my_table`'])

        self.assertEqual(leftover, [])
        self.assertEqual(ast.modifier, EMPTY_NODE)
        self.assertEqual(ast.fields[0].expression, EMPTY_NODE)
        self.assertEqual(ast.fields[0].exception, EMPTY_NODE)
        self.assertEqual(ast.fields[0].replacement, EMPTY_NODE)
        self.assertEqual(ast.from_.first_from[0].path, ('my_project', 'my_dataset', 'my_table'))
        self.assertEqual(ast.from_.first_from[1], EMPTY_NODE)
        self.assertEqual(ast.from_.joins, [])
        self.assertEqual(ast.where, EMPTY_NODE)
        self.assertEqual(ast.group_by, EMPTY_NODE)
        self.assertEqual(ast.having, EMPTY_NODE)

    def test_select_as(self):
        # type: () -> None
        ast, leftover = select(['SELECT', '*', 'FROM', '`my_project.my_dataset.my_table`',
                                'AS', 'TableAlias'])

        self.assertEqual(leftover, [])
        self.assertEqual(ast.from_.first_from[1], 'TableAlias')

    @data(
        dict(
            tokens=['`my_project.my_dataset.my_table`'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias=EMPTY_NODE,
            num_joins=0,
            join_type=EMPTY_NODE,
            join_table_path=EMPTY_NODE,
            join_table_alias=EMPTY_NODE,
            join_conditions=EMPTY_NODE
        ),
        dict(
            tokens=['`my_project.my_dataset.my_table`', 'JOIN',
                    '`my_project.my_dataset.my_table2`', 'ON',
                    'my_table', '.', 'a', '=', 'my_table2', '.', 'b'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias=EMPTY_NODE,
            num_joins=1,
            join_type=EMPTY_NODE,
            join_table_path=('my_project', 'my_dataset', 'my_table2'),
            join_table_alias=EMPTY_NODE,
            join_conditions=BinaryExpression(Field(('my_table', 'a')),
                                             '=',
                                             Field(('my_table2', 'b'))),
        ),
        dict(
            tokens=['`my_project.my_dataset.my_table`', 'JOIN',
                    '`my_project.my_dataset.my_table2`', 'USING', '(', 'a', ',', 'b', ')'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias=EMPTY_NODE,
            num_joins=1,
            join_type=EMPTY_NODE,
            join_table_path=('my_project', 'my_dataset', 'my_table2'),
            join_table_alias=EMPTY_NODE,
            join_conditions=('a', 'b')
        ),
        dict(
            tokens=['`my_project.my_dataset.my_table`', 'JOIN',
                    '`my_project.my_dataset.my_table2`', 'USING', '(', 'a', ')'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias=EMPTY_NODE,
            num_joins=1,
            join_type=EMPTY_NODE,
            join_table_path=('my_project', 'my_dataset', 'my_table2'),
            join_table_alias=EMPTY_NODE,
            join_conditions=('a',)
        ),
        dict(
            tokens=['`my_project.my_dataset.my_table`', 'AS', 'table1', 'JOIN',
                    '`my_project.my_dataset.my_table2`', 'AS', 'table2', 'USING', '(', 'a', ')'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias='table1',
            num_joins=1,
            join_type=EMPTY_NODE,
            join_table_path=('my_project', 'my_dataset', 'my_table2'),
            join_table_alias='table2',
            join_conditions=('a',)
        ),
        dict(
            tokens=['`my_project.my_dataset.my_table`', 'FULL', 'JOIN',
                    '`my_project.my_dataset.my_table2`', 'USING', '(', 'a', ')'],
            first_from_path=('my_project', 'my_dataset', 'my_table'),
            first_from_alias=EMPTY_NODE,
            num_joins=1,
            join_type='FULL',
            join_table_path=('my_project', 'my_dataset', 'my_table2'),
            join_table_alias=EMPTY_NODE,
            join_conditions=('a',)
        ),
    )
    @unpack
    def test_data_source(self, tokens,  # type: List[str]
                         first_from_path,  # type: Tuple[str, ...]
                         first_from_alias,  # type: Union[_EmptyNode, str]
                         num_joins,  # type: int
                         join_type,  # type: Union[_EmptyNode, str]
                         join_table_path,  # type: Union[_EmptyNode, Tuple[str, ...]]
                         join_table_alias,   # type: Union[_EmptyNode, str]
                         join_conditions  # type: Union[_EmptyNode, Tuple[str, ...], Tuple[Field, ...]]  # noqa: E501
                         ):
        # type: (...) -> None
        ast, leftover = data_source(tokens)

        self.assertEqual(leftover, [])
        self.assertEqual(ast.first_from[0].path, first_from_path)
        self.assertEqual(ast.first_from[1], first_from_alias)
        self.assertEqual(len(ast.joins), num_joins)
        if num_joins > 0:
            join = ast.joins[0]
            self.assertEqual(join[0], join_type)
            self.assertEqual(join[1][0].path, join_table_path)
            self.assertEqual(join[1][1], join_table_alias)
            self.assertEqual(repr(join[2]), repr(join_conditions))

    def test_core_expression(self):
        # type: () -> None
        ast, leftover = core_expression(['a'])

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Field))
        self.assertEqual(ast.path, ('a',))

    @data(
        (
            ['SELECT', 'a', 'FROM', '`my_project.my_dataset.my_table`'],
            ('a',)
        ),
        (
            ['SELECT', 'my_table', '.', 'a', 'FROM', '`my_project.my_dataset.my_table`'],
            ('my_table', 'a',)
        )
    )
    @unpack
    def test_select_field(self, tokens, path):
        # type: (List[str], Tuple[str, ...]) -> None
        ast, leftover = select(tokens)

        self.assertEqual(leftover, [])
        field = ast.fields[0].children[0]
        self.assertTrue(isinstance(field, Field))
        self.assertEqual(field.path, path)

    def test_order_by(self):
        # type: () -> None
        ast, leftover = query_expression(['SELECT', '*', 'FROM', 'my_table',
                                          'ORDER', 'BY', 'a', 'ASC'])

        self.assertEqual(leftover, [])
        self.assertEqual(len(ast.order_by), 1)
        (field, direction) = ast.order_by[0]
        self.assertEqual(field.path, ('a',))
        self.assertEqual(direction, 'ASC')

    def test_limit_offset(self):
        # type: () -> None
        ast, leftover = query_expression(['SELECT', '*', 'FROM', 'my_table',
                                          'LIMIT', '5', 'OFFSET', '10'])

        self.assertEqual(leftover, [])
        limit_expression, offset_expression = ast.limit
        self.assertEqual(limit_expression.value, 5)
        self.assertEqual(offset_expression.value, 10)

    def test_group_by(self):
        # type: () -> None
        ast, leftover = select(['SELECT', '*', 'FROM', 'my_table', 'GROUP', 'BY', 'a'])

        self.assertEqual(leftover, [])
        self.assertEqual(len(ast.group_by), 1)
        field = ast.group_by[0]
        self.assertEqual(field.path, ('a',))

    def test_selector_alias(self):
        # type: () -> None
        ast, leftover = select(['SELECT', 'a', 'AS', 'field_name', 'FROM', 'my_table'])

        self.assertEqual(leftover, [])
        self.assertEqual(len(ast.fields), 1)
        selector = ast.fields[0]
        alias = selector.alias
        self.assertEqual(alias, 'field_name')

    @data(
        dict(tokens=['COUNT', '(', '*', ')'],
             countee=Value(1, BQScalarType.INTEGER),
             distinct=False),
        dict(tokens=['COUNT', '(', 'DISTINCT', 'a', ')'],
             countee=Field(('a',)),
             distinct=True),
        dict(tokens=['COUNT', '(', '1', ')'],
             countee=Value(1, BQScalarType.INTEGER),
             distinct=False),
    )
    @unpack
    def test_count(self, tokens,  # type: List[str]
                   countee,  # type: EvaluatableNode
                   distinct,
                   ):
        # type: (...) -> None
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, _AggregatingFunctionCall))
        self.assertEqual(ast.children, [countee])
        self.assertIsInstance(ast.function_info, Count)
        self.assertEqual(ast.function_info.name(), 'COUNT')
        self.assertEqual(ast.function_info.distinct, distinct)

    @data(
        (['a', 'IS', 'NULL'], Field(('a',)), True),
        (['b', 'IS', 'NOT', 'NULL'], Field(('b',)), False)
    )
    @unpack
    def test_null_check(self, tokens,  # type: List[str]
                        expression,  # type: Field
                        direction  # type: bool
                        ):
        # type: (...) -> None
        ast, leftover = post_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, NullCheck))
        self.assertEqual(ast.children[0], expression)
        self.assertEqual(ast.direction, direction)

    @data(
        (['IN'], True),
        (['NOT', 'IN'], False)
    )
    @unpack
    def test_in_check(self, direction,  # type: List[str]
                      bool_direction  # type: bool
                      ):
        # type: (...) -> None
        tokens = ['a'] + direction + ['(', '1', ',', '2', ')']
        expression = Field(('a',))
        elements = [Value(1, BQScalarType.INTEGER), Value(2, BQScalarType.INTEGER)]

        ast, leftover = post_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, InCheck))
        self.assertEqual(ast.children[0], expression)
        self.assertEqual(ast.direction, bool_direction)
        self.assertEqual(ast.children[1:], elements)

    def test_if(self):
        # IF (3 < 5, 0, 1)
        tokens = ['IF', '(', '3', '<', '5', ',', '0', ',', '1', ')']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, If))
        condition, then, else_ = ast.children
        left, right = condition.children
        self.assertEqual(left, Value(3, BQScalarType.INTEGER))
        self.assertEqual(condition.operator_info.operator, '<')
        self.assertEqual(right, Value(5, BQScalarType.INTEGER))
        self.assertEqual(then, Value(0, BQScalarType.INTEGER))
        self.assertEqual(else_, Value(1, BQScalarType.INTEGER))

    def test_not(self):
        tokens = ['NOT', '2', '=', '2']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Not))
        left, right = ast.children[0].children
        self.assertEqual(left, Value(2, BQScalarType.INTEGER))
        self.assertEqual(ast.children[0].operator_info.operator, '=')
        self.assertEqual(right, Value(2, BQScalarType.INTEGER))

    def test_unary_negation(self):
        tokens = ['-', '2']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, UnaryNegation))
        self.assertEqual(ast.children[0], Value(2, BQScalarType.INTEGER))

    def test_case(self):
        tokens = ['CASE', 'WHEN', 'a', '=', '1', 'THEN', '1',
                  'WHEN', 'a', '=', '2', 'THEN', '2',
                  'ELSE', '3', 'END']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Case))

        first_when, first_then, second_when, second_then, else_ = ast.children
        self.assertEqual(first_when.children[0], Field(('a',)))
        self.assertEqual(first_when.operator_info.operator, '=')
        self.assertEqual(first_when.children[1], Value(1, BQScalarType.INTEGER))
        self.assertEqual(first_then, Value(1, BQScalarType.INTEGER))

        self.assertEqual(second_when.children[0], Field(('a',)))
        self.assertEqual(second_when.operator_info.operator, '=')
        self.assertEqual(second_when.children[1], Value(2, BQScalarType.INTEGER))
        self.assertEqual(second_then, Value(2, BQScalarType.INTEGER))

        self.assertEqual(else_, Value(3, BQScalarType.INTEGER))

    def test_cast(self):
        tokens = ['CAST', '(', '1', 'AS', 'STRING', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Cast))
        self.assertEqual(ast.children[0], Value(1, BQScalarType.INTEGER))
        self.assertEqual(ast.type_, BQScalarType.STRING)

    def test_exists(self):
        tokens = ['EXISTS', '(', 'SELECT', '*', 'FROM', 'Table', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Exists))
        self.assertTrue(isinstance(ast.select, Select))
        self.assertEqual(ast.select.from_.first_from[0].path, ('Table',))

    def test_extract(self):
        tokens = ['EXTRACT', '(', 'DAY', 'FROM', 'date_field', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        self.assertTrue(isinstance(ast, Extract))
        self.assertEqual(ast.part, 'DAY')
        self.assertEqual(ast.children[0].path, ('date_field',))


if __name__ == '__main__':
    unittest.main()
