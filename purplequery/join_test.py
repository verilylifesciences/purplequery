# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from typing import List, Tuple, Type, Union  # noqa: F401

import pandas as pd
from ddt import data, ddt, unpack

from purplequery.bq_abstract_syntax_tree import (EMPTY_NODE, AbstractSyntaxTreeNode,  # noqa: F401
                                                 DatasetTableContext, EvaluationContext, _EmptyNode)
from purplequery.bq_types import BQScalarType, TypedDataFrame
from purplequery.dataframe_node import TableReference
from purplequery.grammar import data_source
from purplequery.join import ConditionsType  # noqa: F401
from purplequery.join import DataSource, Join
from purplequery.query_helper import apply_rule
from purplequery.tokenizer import tokenize


@ddt
class JoinTest(unittest.TestCase):

    def setUp(self):
        # type: () -> None
        self.table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1], [2]], columns=['a']),
                        types=[BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })

    def test_data_source(self):
        data_source = DataSource((TableReference(('my_project', 'my_dataset', 'my_table')),
                                  EMPTY_NODE), [])
        data_source_context = data_source.create_context(self.table_context)

        self.assertEqual(data_source_context.table.to_list_of_lists(), [[1], [2]])
        self.assertEqual(list(data_source_context.table.dataframe), ['my_table.a'])
        self.assertEqual(data_source_context.table.types, [BQScalarType.INTEGER])

    @data(
        dict(
            join_type='JOIN',
            table1=[[1, 9]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 9, 1, 2]]),
        dict(
            join_type='INNER JOIN',
            table1=[[1, 9]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 9, 1, 2]]),
        dict(
            join_type='left join',
            table1=[[1, 4], [2, 5], [3, 6]],
            table2=[[1, 3], [2, 4]],
            result=[[1, 4, 1, 3], [2, 5, 2, 4], [3, 6, None, None]]),
        dict(
            join_type='LEFT OUTER JOIN',
            table1=[[1, 4], [2, 5], [3, 6]],
            table2=[[1, 3], [2, 4]],
            result=[[1, 4, 1, 3], [2, 5, 2, 4], [3, 6, None, None]]),
        dict(
            join_type='RIGHT JOIN',
            table1=[[1, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 5, 1, 2], [None, None, 3, 4]]),
        dict(
            join_type='RIGHT OUTER JOIN',
            table1=[[1, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 5, 1, 2], [None, None, 3, 4]]),
        dict(
            join_type='FULL JOIN',
            table1=[[1, 3], [2, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 3, 1, 2], [2, 5, None, None], [None, None, 3, 4]]),
        dict(
            join_type='FULL OUTER JOIN',
            table1=[[1, 3], [2, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 3, 1, 2], [2, 5, None, None], [None, None, 3, 4]]),
        dict(
            join_type='CROSS JOIN',
            table1=[[1, 3], [2, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 3, 1, 2],
                    [1, 3, 3, 4],
                    [2, 5, 1, 2],
                    [2, 5, 3, 4]]),
        dict(
            join_type=',',
            table1=[[1, 3], [2, 5]],
            table2=[[1, 2], [3, 4]],
            result=[[1, 3, 1, 2],
                    [1, 3, 3, 4],
                    [2, 5, 1, 2],
                    [2, 5, 3, 4]]),
    )
    @unpack
    def test_data_source_joins(self, join_type,  # type: Union[_EmptyNode, str]
                               table1,  # type: List[List[int]]
                               table2,  # type: List[List[int]]
                               result  # type: List[List[int]]
                               ):
        # type: (...) -> None
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame(table1, columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame(table2, columns=['a', 'c']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })
        tokens = tokenize('my_table {} my_table2 {}'.format(
                join_type, 'USING (a)' if join_type not in (',', 'CROSS JOIN') else ''))
        data_source_node, leftover = apply_rule(data_source, tokens)
        self.assertFalse(leftover)
        assert isinstance(data_source_node, DataSource)
        context = data_source_node.create_context(table_context)

        self.assertEqual(context.table.to_list_of_lists(), result)
        self.assertEqual(list(context.table.dataframe),
                         ['my_table.a', 'my_table.b', 'my_table2.a', 'my_table2.c'])

    @data(
        dict(
            join_type='INNER JOIN',
            result=[[1, 2]]),
        dict(
            join_type='CROSS JOIN',  # With an on clause, CROSS functions like inner.
            result=[[1, 2]]),
        dict(
            join_type='LEFT OUTER JOIN',
            result=[[1, 2], [2, None]]),
        dict(
            join_type='RIGHT OUTER JOIN',
            result=[[1, 2], [None, 0]]),
        dict(
            join_type='FULL OUTER JOIN',
            result=[[1, 2], [2, None], [None, 0]]),
    )
    @unpack
    def test_data_source_join_on_arbitrary_bool(self, join_type,  # type: Union[_EmptyNode, str]
                                                result  # type: List[List[int]]
                                                ):
        # type: (...) -> None
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1], [2]], columns=['a']),
                        types=[BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[2], [0]], columns=['b']),
                        types=[BQScalarType.INTEGER]
                    )
                }
            }
        })
        tokens = tokenize('my_table {} my_table2 ON MOD(a + b, 3) = 0'.format(join_type))
        data_source_node, leftover = apply_rule(data_source, tokens)
        self.assertFalse(leftover)
        assert isinstance(data_source_node, DataSource)
        context = data_source_node.create_context(table_context)

        self.assertEqual(context.table.to_list_of_lists(), result)

    @data(
        dict(
            condition='my_table2.c = my_table.a',
            expected_result=[[1, 9, 1, 2]]
        ),
        dict(
            condition='my_table.a = my_table2.d',
            expected_result=[[2, 8, 1, 2], [2, 1, 1, 2]]
        ),
        dict(
            condition='my_table.a = my_table2.d and my_table2.c = my_table.b',
            expected_result=[[2, 1, 1, 2]]
        ),
    )
    @unpack
    def test_data_source_join_on_field_comparison(self, condition, expected_result):
        # type: (str, List[List[int]]) -> None
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 9], [2, 8], [2, 1]], columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 2], [3, 4]], columns=['c', 'd']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })
        data_source_node, leftover = data_source(
            tokenize(
                'my_project.my_dataset.my_table join my_project.my_dataset.my_table2 on {}'.format(
                        condition)))
        self.assertFalse(leftover)
        assert isinstance(data_source_node, DataSource)
        context = data_source_node.create_context(table_context)

        self.assertEqual(context.table.to_list_of_lists(), expected_result)
        self.assertEqual(list(context.table.dataframe),
                         ['my_table.a', 'my_table.b', 'my_table2.c', 'my_table2.d'])

    def test_data_source_join_overlapping_fields(self):
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 9]], columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'd']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })
        initial_table = (TableReference(('my_project', 'my_dataset', 'my_table')), EMPTY_NODE)
        join_type = 'INNER'
        join_table = (TableReference(('my_project', 'my_dataset', 'my_table2')), EMPTY_NODE)
        join_on = EMPTY_NODE
        joins = [(join_type, join_table, join_on)]

        data_source = DataSource(initial_table, joins)
        context = data_source.create_context(table_context)

        self.assertEqual(context.table.to_list_of_lists(), [[1, 9, 1, 2]])
        self.assertEqual(list(context.table.dataframe),
                         ['my_table.a', 'my_table.b', 'my_table2.a', 'my_table2.d'])

    def test_data_source_join_multiple_columns(self):
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 2, 3], [1, 5, 6]], columns=['a', 'b', 'c']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 2, 7], [3, 2, 8]], columns=['a', 'b', 'd']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })
        initial_table = (TableReference(('my_project', 'my_dataset', 'my_table')), EMPTY_NODE)
        join_type = 'FULL'
        join_table = (TableReference(('my_project', 'my_dataset', 'my_table2')), EMPTY_NODE)
        join_on = ('a', 'b')
        joins = [(join_type, join_table, join_on)]

        data_source = DataSource(initial_table, joins)
        context = data_source.create_context(table_context)

        result = [
            [1, 2, 3, 1, 2, 7],
            [1, 5, 6, None, None, None],
            [None, None, None, 3, 2, 8]
        ]
        self.assertEqual(context.table.to_list_of_lists(), result)

    def test_data_source_join_multiple_joins(self):
        table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b1', 'c1']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 8, 9], [0, 7, 2]], columns=['a', 'b', 'c2']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table3': TypedDataFrame(
                        pd.DataFrame([[3, 4, 5], [6, 7, 8]], columns=['a3', 'b', 'c3']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        })
        initial_table = (TableReference(('my_project', 'my_dataset', 'my_table')), EMPTY_NODE)
        join_type = 'FULL'
        join_table2 = (TableReference(('my_project', 'my_dataset', 'my_table2')), EMPTY_NODE)
        join_table3 = (TableReference(('my_project', 'my_dataset', 'my_table3')), EMPTY_NODE)
        joins = [(join_type, join_table2, ('a',)),
                 (join_type, join_table3, ('b',))]
        data_source = DataSource(initial_table, joins)
        context = data_source.create_context(table_context)

        result = [
            [1, 2, 3, 1, 8, 9, None, None, None],
            [4, 5, 6, None, None, None, None, None, None],
            [None, None, None, 0, 7, 2, 6, 7, 8],
            [None, None, None, None, None, None, 3, 4, 5]
        ]
        self.assertEqual(context.table.to_list_of_lists(), result)

    @data(
        dict(
            join_type='fake_join_type',
            join_on=['a'],
            error_type=NotImplementedError,
            error="Join type FAKE_JOIN_TYPE is not supported"
        ),
        dict(
            join_type='INNER',
            join_on=('b',),
            error_type=ValueError,
            error=(r"JOIN USING key must exist in exactly two tables; "
                   r"exists in these: \['my_table2'\]")
        ),
    )
    @unpack
    def test_data_source_join_error(self, join_type,  # type: str
                                    join_on,  # type: ConditionsType
                                    error_type,  # type: Type[BaseException]
                                    error  # type: str
                                    ):
        # type: (...) -> None
        initial_table = (TableReference(('my_project', 'my_dataset', 'my_table')), EMPTY_NODE)
        join_table = (TableReference(('my_project', 'my_dataset', 'my_table2')), EMPTY_NODE)
        joins = [Join(join_type, join_table, join_on)]

        data_source = DataSource(initial_table, joins)
        with self.assertRaisesRegexp(error_type, error):
            data_source.create_context(self.table_context)


if __name__ == '__main__':
    unittest.main()
