# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from typing import List, Tuple  # noqa: F401

import pandas as pd
from ddt import data, ddt, unpack

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import (EMPTY_NODE, DatasetTableContext, EvaluatableNode,  # noqa: F401
                                     Field)
from bq_types import BQScalarType, TypedDataFrame
from dataframe_node import QueryExpression, Select, TableReference
from evaluatable_node import Selector, StarSelector, Value
from grammar import query_expression as query_expression_rule
from grammar import select as select_rule
from join import DataSource
from tokenizer import tokenize


@ddt
class DataframeNodeTest(unittest.TestCase):

    def setUp(self):
        # type: () -> None
        self.table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame([[1], [2], [3]], columns=['a']), types=[BQScalarType.INTEGER])}}})

    def test_marker_syntax_tree_node(self):
        # type: () -> None
        self.assertEqual(Select.literal(), 'SELECT')

    def test_query_expression(self):
        # type: () -> None
        from_ = DataSource((TableReference(('my_project', 'my_dataset', 'my_table')),
                            EMPTY_NODE), [])
        selector = StarSelector(EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)
        select = Select(EMPTY_NODE, [selector], from_, EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)
        qe = QueryExpression(EMPTY_NODE, select, EMPTY_NODE, EMPTY_NODE)
        dataframe, table_name = qe.get_dataframe(self.table_context)

        self.assertEqual(table_name, None)
        self.assertEqual(dataframe.to_list_of_lists(), [[1], [2], [3]])
        self.assertEqual(list(dataframe.dataframe), ['my_table.a'])
        self.assertEqual(dataframe.types, [BQScalarType.INTEGER])

    def test_query_expression_limit(self):
        # type: () -> None
        from_ = DataSource((TableReference(('my_project', 'my_dataset', 'my_table')),
                            EMPTY_NODE), [])
        selector = StarSelector(EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)
        select = Select(EMPTY_NODE, [selector], from_, EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)

        limit = Value(1, BQScalarType.INTEGER)
        offset = Value(1, BQScalarType.INTEGER)
        qe = QueryExpression(EMPTY_NODE, select, EMPTY_NODE, (limit, offset))
        dataframe, table_name = qe.get_dataframe(self.table_context)

        self.assertEqual(dataframe.to_list_of_lists(), [[2]])

    @data(
        dict(query_expression='select a as d from my_table union all select c as d from my_table',
             expected_result=[[1], [3]]),
        dict(query_expression='select 1 union all select 2.0',
             expected_result=[[1.0], [2.0]]),
    )
    @unpack
    def test_query_expression_set_operation(self, query_expression, expected_result):
        table_context = DatasetTableContext(
            {'my_project':
             {'my_dataset':
              {'my_table':
               TypedDataFrame(pd.DataFrame([[1, 2, 3]],
                                           columns=['a', 'b', 'c']),
                              [BQScalarType.INTEGER,
                               BQScalarType.INTEGER,
                               BQScalarType.INTEGER])}}})
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        dataframe, unused_table_name = query_expression_node.get_dataframe(table_context)
        self.assertFalse(leftover)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data(
        dict(query_expression='select 1 union all select 2, 3',
             error='mismatched column count: 1 vs 2'),
        dict(query_expression='select 1 union all select "foo"',
             error='Cannot implicitly coerce the given types'),
    )
    @unpack
    def test_query_expression_set_operation_error(self, query_expression, error):
        table_context = DatasetTableContext(
            {'my_project':
             {'my_dataset':
              {'my_table':
               TypedDataFrame(pd.DataFrame([[1, 2, 3]],
                                           columns=['a', 'b', 'c']),
                              [BQScalarType.INTEGER,
                               BQScalarType.INTEGER,
                               BQScalarType.INTEGER])}}})
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        self.assertFalse(leftover)
        with self.assertRaisesRegexp(ValueError, error):
            query_expression_node.get_dataframe(table_context)

    @data(
        dict(
            query_expression='WITH q1 as (SELECT a+1 as b FROM my_table) SELECT * from q1',
            expected_result=[[2], [3], [4]]),
        # This example is closely based on
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#with-clause
        dict(query_expression=(
            'WITH q1 AS (SELECT a+1 as b FROM my_table LIMIT 1)'  # q1 is [[2]]
            'SELECT *'
            'FROM'
            '  (WITH q2 AS (SELECT b+1 as c FROM q1),'  # q1 resolves to [[2]]
            '        q3 AS (SELECT b+2 as d FROM q1),'  # q1 resolves to [[2]]
            '        q1 AS (SELECT b+4 as e FROM q1),'  # q1 (in the query) resolves to [[2]]
            '        q4 AS (SELECT e+8 as f FROM q1)'   # q1 resolves to the WITH subquery
                                                        # on the previous line.
            '   SELECT e, c, d, f FROM q1, q2, q3, q4)'),  # q1 resolves to 3rd inner WITH subquery.
            expected_result=[[6, 3, 4, 14]],
        ),
    )
    @unpack
    def test_with_clause(self, query_expression, expected_result):
        # type: (str, List[List[int]]) -> None
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        self.assertFalse(leftover)
        assert isinstance(query_expression_node, QueryExpression)
        dataframe, _ = query_expression_node.get_dataframe(self.table_context)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data(
        dict(
            query_expression=(
                'WITH q1 as (SELECT a+1 as b FROM my_table),'
                'q1 as (SELECT a+2 as b FROM my_table)'
                'SELECT * from q1'),
            error='Duplicate names in WITH clauses are not allowed'),
    )
    @unpack
    def test_with_clause_error(self, query_expression, error):
        # type: (str, str) -> None
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        self.assertFalse(leftover)
        assert isinstance(query_expression_node, QueryExpression)
        with self.assertRaisesRegexp(ValueError, error):
            query_expression_node.get_dataframe(self.table_context)

    # constants for use in data-driven syntax tests below
    with_clause = 'with foo as (select 1), bar as (select 2)'
    select_clause = 'select 3'
    order_by_clause = 'order by a, b'
    limit_clause = 'limit 5'

    @data(
        # core query expression grammar (no recursion)
        dict(query=(with_clause, select_clause, order_by_clause, limit_clause)),
        dict(query=(with_clause, select_clause, limit_clause)),
        dict(query=(with_clause, select_clause, order_by_clause)),
        dict(query=(with_clause, select_clause)),
        dict(query=(select_clause, order_by_clause, limit_clause)),
        dict(query=(select_clause, limit_clause)),
        dict(query=(select_clause, order_by_clause)),
        dict(query=(select_clause,)),

        # parenthesized recursion
        dict(query=(with_clause,
                    '(', with_clause, select_clause, order_by_clause, limit_clause, ')',
                    order_by_clause, limit_clause)),
        dict(query=(with_clause,
                    '(', with_clause, select_clause, order_by_clause, limit_clause, ')',
                    limit_clause)),
        dict(query=(with_clause,
                    '(', with_clause, select_clause, order_by_clause, limit_clause, ')',
                    order_by_clause)),
        dict(query=(with_clause,
                    '(', with_clause, select_clause, order_by_clause, limit_clause, ')')),

        # set operations
        dict(query=(select_clause, 'UNION ALL', select_clause)),  # simplest case
        dict(query=(
                # first query expression
                with_clause, select_clause, order_by_clause, limit_clause,
                'UNION ALL',
                # second query expression
                with_clause, select_clause, order_by_clause, limit_clause)),
    )
    @unpack
    def test_query_expression_syntax(self, query):
        query_expression = ' '.join(query)
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        self.assertFalse(leftover)

    @data(
        dict(query_expression='select * from my_table order by a DESC',
             expected_result=[[3], [2], [1]]),
        dict(query_expression='select a from my_table order by 1 DESC',
             expected_result=[[3], [2], [1]]),
    )
    @unpack
    def test_query_expression_order_by(self, query_expression, expected_result):
        # type: (str, List[List[int]]) -> None
        query_expression_node, leftover = query_expression_rule(tokenize(query_expression))
        assert isinstance(query_expression_node, QueryExpression)
        dataframe, unused_table_name = query_expression_node.get_dataframe(self.table_context)
        self.assertFalse(leftover)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    def test_select(self):
        # type: () -> None
        from_ = DataSource((TableReference(('my_project', 'my_dataset', 'my_table')),
                            EMPTY_NODE), [])
        selector = StarSelector(EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)
        select = Select(EMPTY_NODE, [selector], from_, EMPTY_NODE, EMPTY_NODE, EMPTY_NODE)
        dataframe, table_name = select.get_dataframe(self.table_context)

        self.assertEqual(table_name, None)
        self.assertEqual(dataframe.to_list_of_lists(), [[1], [2], [3]])
        self.assertEqual(list(dataframe.dataframe), ['my_table.a'])
        self.assertEqual(dataframe.types, [BQScalarType.INTEGER])

    @data(
        dict(select='select c from my_table group by c', expected_result=[[3]]),
        dict(select='select my_table.c from my_table group by c', expected_result=[[3]]),
        dict(select='select c from my_table group by my_table.c', expected_result=[[3]]),
        dict(select='select my_table.c from my_table group by my_table.c', expected_result=[[3]]),
        dict(select='select c from my_table group by 1', expected_result=[[3]]),
        dict(select='select max(b), c from my_table group by 2', expected_result=[[3, 3]]),
        dict(select='select c+1 from my_table group by c', expected_result=[[4]]),
        dict(select='select c+1 from my_table group by 1', expected_result=[[4]]),
        dict(select='select c+1 from my_table group by c, 1', expected_result=[[4]]),
        dict(select='select c+1 as d from my_table group by c,d', expected_result=[[4]]),
        dict(select='select c+1 as d from my_table group by c,1', expected_result=[[4]]),
        dict(select='select c+1 from my_table group by c,1', expected_result=[[4]]),
        # Naively evaluating the below after the group by has occurred fails with
        # Column(s) my_table.c already selected
        dict(select='select c+1 as d, count(*) from my_table group by d', expected_result=[[4, 2]]),
        # Naively evaluating the below after the group by has occurred tries and fails to add
        # two SeriesGroupbys
        dict(select='select a+c as d from my_table group by d', expected_result=[[4]]),
    )
    @unpack
    def test_select_group_by(self, select, expected_result):
        # type: (str, List[List[int]]) -> None
        group_table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame(
                    [[1, 2, 3], [1, 3, 3]],
                    columns=['a', 'b', 'c']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
            )}}})
        select_node, leftover = select_rule(tokenize(select))
        assert isinstance(select_node, Select)
        dataframe, unused_table_name = select_node.get_dataframe(group_table_context)
        self.assertFalse(leftover)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data(
        dict(select='select b from my_table group by c'),
        dict(select='select b,c from my_table group by c'),
        # Note: a is actually constant within the group, but it doesn't matter because you can't
        # tell that statically from the query.
        dict(select='select a from my_table group by c'),
    )
    @unpack
    def test_select_group_by_error(self, select):
        # type: (str) -> None
        group_table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame(
                    [[1, 2, 3], [1, 3, 3]],
                    columns=['a', 'b', 'c']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
            )}}})
        select_node, leftover = select_rule(tokenize(select))
        assert isinstance(select_node, Select)
        self.assertFalse(leftover)
        with self.assertRaisesRegexp(ValueError, "not aggregated or grouped by"):
            select_node.get_dataframe(group_table_context)

    @data(
        dict(select='select distinct a from my_table', expected_result=[[1]]),
        dict(select='select distinct b from my_table', expected_result=[[2], [3]]),
        dict(select='select distinct a, b from my_table', expected_result=[[1, 2], [1, 3]]),
    )
    @unpack
    def test_select_distinct(self, select, expected_result):
        # type: (str, List[List[int]]) -> None
        table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame(
                    [[1, 2], [1, 3]],
                    columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
            )}}})
        select_node, leftover = select_rule(tokenize(select))
        assert isinstance(select_node, Select)
        dataframe, unused_table_name = select_node.get_dataframe(table_context)
        self.assertFalse(leftover)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data(
        # WHERE b = 4
        (BinaryExpression(Field(('b',)), '=', Value(value=4, type_=BQScalarType.INTEGER)),),
        # WHERE b = 4 AND a = 3
        (BinaryExpression(
            BinaryExpression(Field(('b',)), '=', Value(value=4, type_=BQScalarType.INTEGER)),
            'AND',
            BinaryExpression(Field(('a',)), '=', Value(value=3, type_=BQScalarType.INTEGER))),)
    )
    @unpack
    def test_select_where(self, where):
        # type: (EvaluatableNode) -> None
        where_table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame(
                    [[1, 2], [3, 4]],
                    columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
            )}}})

        fields = [Selector(Field(('a',)), EMPTY_NODE)]
        from_ = DataSource((TableReference(('my_project', 'my_dataset', 'my_table')),
                            EMPTY_NODE), [])
        select = Select(EMPTY_NODE, fields, from_, where, EMPTY_NODE, EMPTY_NODE)
        dataframe, table_name = select.get_dataframe(where_table_context)

        self.assertEqual(dataframe.to_list_of_lists(), [[3]])

    @data(
        dict(select='SELECT sum(a) as c FROM my_table GROUP BY b HAVING c > 4',
             expected_result=[[7]]),
        dict(select='SELECT b FROM my_table GROUP BY b HAVING sum(a) > 4',
             expected_result=[[3]]),
        dict(select='SELECT sum(a) as c FROM my_table GROUP BY b HAVING b > 4',
             expected_result=[[4]]),
        dict(select='SELECT sum(a) as c FROM my_table GROUP BY b HAVING b=1 AND c<0 AND MIN(a)<0',
             expected_result=[[-98]]),
    )
    @unpack
    def test_select_having(self, select, expected_result):
        # type: (str, List[List[int]]) -> None
        group_table_context = DatasetTableContext(
            {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame(
                    [[1, 3], [6, 3], [4, 5], [-100, 1], [2, 1]],
                    columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
            )}}})

        select_node, leftover = select_rule(tokenize(select))
        assert isinstance(select_node, Select)
        dataframe, unused_table_name = select_node.get_dataframe(group_table_context)
        self.assertFalse(leftover)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data((('my_project', 'my_dataset', 'my_table'),),
          (('my_dataset', 'my_table'),),
          (('my_table',),),
          (('my_project.my_dataset.my_table',),))
    @unpack
    def test_table_reference(self, reference):
        # type: (Tuple[str, ...]) -> None
        table_ref = TableReference(reference)
        dataframe, table_name = table_ref.get_dataframe(self.table_context)

        self.assertEqual(table_name, 'my_table')
        self.assertEqual(dataframe.to_list_of_lists(), [[1], [2], [3]])
        self.assertEqual(list(dataframe.dataframe), ['a'])
        self.assertEqual(dataframe.types, [BQScalarType.INTEGER])

    def test_table_reference_multi_project(self):
        # type: () -> None
        new_table_context = DatasetTableContext({
            'project1': {
                'dataset1': {'table1': TypedDataFrame(pd.DataFrame(), [])}
            },
            'project2': {
                'dataset2': {'table2': TypedDataFrame(pd.DataFrame(), [])}
            }
        })
        table_ref = TableReference(('dataset1', 'table1'))
        expected_error = "Non-fully-qualified table \\('dataset1', 'table1'\\) with multiple "\
            "possible projects \\['project1', 'project2'\\]"
        with self.assertRaisesRegexp(ValueError, expected_error):
            table_ref.get_dataframe(new_table_context)

    def test_table_reference_multi_dataset(self):
        # type: () -> None
        new_table_context = DatasetTableContext({
            'project1': {
                'dataset1': {'table1': TypedDataFrame(pd.DataFrame(), [])},
                'dataset2': {'table2': TypedDataFrame(pd.DataFrame(), [])}
            },
        })
        table_ref = TableReference(('table1',))
        expected_error = "Non-fully-qualified table \\('table1',\\) with multiple possible "\
            "datasets \\['dataset1', 'dataset2'\\]"
        with self.assertRaisesRegexp(ValueError, expected_error):
            table_ref.get_dataframe(new_table_context)


if __name__ == '__main__':
    unittest.main()
