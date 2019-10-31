# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from typing import List, Tuple, Union  # noqa: F401

from ddt import data, ddt, unpack

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import EMPTY_NODE, EvaluatableNode, Field, _EmptyNode  # noqa: F401
from bq_types import BQScalarType
from dataframe_node import DataSource, QueryExpression, Select, TableReference
from evaluatable_node import (Case, Cast, Count, Exists, Extract, If, InCheck, Not, NullCheck,
                              Selector, StarSelector, UnaryNegation, Value,
                              _AggregatingFunctionCall)
from grammar import core_expression, data_source, post_expression, query_expression, select


@ddt
class GrammarTest(unittest.TestCase):

    def test_query_expression(self):
        # type: () -> None
        ast, leftover = query_expression(
            ['SELECT', '*', 'FROM', '`my_project.my_dataset.my_table`'])
        self.assertEqual(leftover, [])

        assert isinstance(ast, QueryExpression)
        self.assertEqual(ast.base_query.__class__, Select)

    def test_select(self):
        # type: () -> None
        ast, leftover = select(['SELECT', '*', 'FROM', '`my_project.my_dataset.my_table`'])

        self.assertEqual(leftover, [])
        assert isinstance(ast, Select)
        self.assertEqual(ast.modifier, EMPTY_NODE)
        assert isinstance(ast.fields[0], StarSelector)
        self.assertEqual(ast.fields[0].expression, EMPTY_NODE)
        self.assertEqual(ast.fields[0].exception, EMPTY_NODE)
        self.assertEqual(ast.fields[0].replacement, EMPTY_NODE)
        assert isinstance(ast.from_, DataSource)
        assert isinstance(ast.from_.first_from[0], TableReference)
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
        assert isinstance(ast, Select)
        assert isinstance(ast.from_, DataSource)
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

        assert isinstance(ast, DataSource)
        self.assertEqual(leftover, [])
        assert isinstance(ast.first_from[0], TableReference)
        self.assertEqual(ast.first_from[0].path, first_from_path)
        self.assertEqual(ast.first_from[1], first_from_alias)
        self.assertEqual(len(ast.joins), num_joins)
        if num_joins > 0:
            join = ast.joins[0]
            self.assertEqual(join[0], join_type)
            assert isinstance(join[1][0], TableReference)
            self.assertEqual(join[1][0].path, join_table_path)
            self.assertEqual(join[1][1], join_table_alias)
            self.assertEqual(repr(join[2]), repr(join_conditions))

    def test_core_expression(self):
        # type: () -> None
        ast, leftover = core_expression(['a'])

        self.assertEqual(leftover, [])
        assert isinstance(ast, Field)
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
        assert isinstance(ast, Select)
        assert isinstance(ast.fields[0], Selector)
        field = ast.fields[0].children[0]
        assert isinstance(field, Field)
        self.assertEqual(field.path, path)

    def test_order_by(self):
        # type: () -> None
        ast, leftover = query_expression(['SELECT', '*', 'FROM', 'my_table',
                                          'ORDER', 'BY', 'a', 'ASC'])

        assert isinstance(ast, QueryExpression)
        self.assertEqual(leftover, [])
        assert not isinstance(ast.order_by, _EmptyNode)
        self.assertEqual(len(ast.order_by), 1)
        (field, direction) = ast.order_by[0]
        self.assertEqual(field.path, ('a',))
        self.assertEqual(direction, 'ASC')

    def test_limit_offset(self):
        # type: () -> None
        ast, leftover = query_expression(['SELECT', '*', 'FROM', 'my_table',
                                          'LIMIT', '5', 'OFFSET', '10'])
        assert isinstance(ast, QueryExpression)
        self.assertEqual(leftover, [])
        assert not isinstance(ast.limit, _EmptyNode)
        limit_expression, offset_expression = ast.limit
        assert isinstance(limit_expression, Value)
        self.assertEqual(limit_expression.value, 5)
        assert isinstance(offset_expression, Value)
        self.assertEqual(offset_expression.value, 10)

    def test_group_by(self):
        # type: () -> None
        ast, leftover = select(['SELECT', '*', 'FROM', 'my_table', 'GROUP', 'BY', 'a'])

        self.assertEqual(leftover, [])
        assert isinstance(ast, Select)
        assert not isinstance(ast.group_by, _EmptyNode)
        self.assertEqual(len(ast.group_by), 1)
        field = ast.group_by[0]
        assert isinstance(field, Field)
        self.assertEqual(field.path, ('a',))

    def test_selector_alias(self):
        # type: () -> None
        ast, leftover = select(['SELECT', 'a', 'AS', 'field_name', 'FROM', 'my_table'])

        self.assertEqual(leftover, [])
        assert isinstance(ast, Select)
        self.assertEqual(len(ast.fields), 1)
        selector = ast.fields[0]
        assert isinstance(selector, Selector)
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
        assert isinstance(ast, _AggregatingFunctionCall)
        self.assertEqual(ast.children, [countee])
        assert isinstance(ast.function_info, Count)
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
        assert isinstance(ast, NullCheck)
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
        assert isinstance(ast, InCheck)
        self.assertEqual(ast.children[0], expression)
        self.assertEqual(ast.direction, bool_direction)
        self.assertEqual(ast.children[1:], elements)

    def test_if(self):
        # type: () -> None
        # IF (3 < 5, 0, 1)
        tokens = ['IF', '(', '3', '<', '5', ',', '0', ',', '1', ')']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, If)
        condition, then, else_ = ast.children
        assert isinstance(condition, BinaryExpression)
        left, right = condition.children
        self.assertEqual(left, Value(3, BQScalarType.INTEGER))
        self.assertEqual(condition.operator_info.operator, '<')
        self.assertEqual(right, Value(5, BQScalarType.INTEGER))
        self.assertEqual(then, Value(0, BQScalarType.INTEGER))
        self.assertEqual(else_, Value(1, BQScalarType.INTEGER))

    def test_not(self):
        # type: () -> None
        tokens = ['NOT', '2', '=', '2']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, Not)
        assert isinstance(ast.children[0], BinaryExpression)
        left, right = ast.children[0].children
        self.assertEqual(left, Value(2, BQScalarType.INTEGER))
        self.assertEqual(ast.children[0].operator_info.operator, '=')
        self.assertEqual(right, Value(2, BQScalarType.INTEGER))

    def test_unary_negation(self):
        # type: () -> None
        tokens = ['-', '2']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, UnaryNegation)
        self.assertEqual(ast.children[0], Value(2, BQScalarType.INTEGER))

    def test_case(self):
        # type: () -> None
        tokens = ['CASE', 'WHEN', 'a', '=', '1', 'THEN', '1',
                  'WHEN', 'a', '=', '2', 'THEN', '2',
                  'ELSE', '3', 'END']

        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, Case)

        first_when, first_then, second_when, second_then, else_ = ast.children
        assert isinstance(first_when, BinaryExpression)
        self.assertEqual(first_when.children[0], Field(('a',)))
        self.assertEqual(first_when.operator_info.operator, '=')
        self.assertEqual(first_when.children[1], Value(1, BQScalarType.INTEGER))
        self.assertEqual(first_then, Value(1, BQScalarType.INTEGER))

        assert isinstance(second_when, BinaryExpression)
        self.assertEqual(second_when.children[0], Field(('a',)))
        self.assertEqual(second_when.operator_info.operator, '=')
        self.assertEqual(second_when.children[1], Value(2, BQScalarType.INTEGER))
        self.assertEqual(second_then, Value(2, BQScalarType.INTEGER))

        self.assertEqual(else_, Value(3, BQScalarType.INTEGER))

    def test_cast(self):
        # type: () -> None
        tokens = ['CAST', '(', '1', 'AS', 'STRING', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, Cast)
        self.assertEqual(ast.children[0], Value(1, BQScalarType.INTEGER))
        self.assertEqual(ast.type_, BQScalarType.STRING)

    def test_exists(self):
        # type: () -> None
        tokens = ['EXISTS', '(', 'SELECT', '*', 'FROM', 'Table', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, Exists)
        assert isinstance(ast.subquery, QueryExpression)
        assert isinstance(ast.subquery.base_query, Select)
        assert isinstance(ast.subquery.base_query.from_, DataSource)
        assert isinstance(ast.subquery.base_query.from_.first_from[0], TableReference)
        self.assertEqual(ast.subquery.base_query.from_.first_from[0].path, ('Table',))

    def test_extract(self):
        # type: () -> None
        tokens = ['EXTRACT', '(', 'DAY', 'FROM', 'date_field', ')']
        ast, leftover = core_expression(tokens)

        self.assertEqual(leftover, [])
        assert isinstance(ast, Extract)
        self.assertEqual(ast.part, 'DAY')
        assert isinstance(ast.children[0], Field)
        self.assertEqual(ast.children[0].path, ('date_field',))


if __name__ == '__main__':
    unittest.main()
