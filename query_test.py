# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
import unittest
from typing import Any  # noqa: F401

import pandas as pd
from ddt import data, ddt, unpack

import query
from bq_abstract_syntax_tree import EMPTY_CONTEXT, EvaluatableNode
from bq_types import BQScalarType, TypedDataFrame, TypedSeries
from grammar import expression as expression_rule


@ddt
class QueryTest(unittest.TestCase):

    def setUp(self):
        ten_rows = TypedDataFrame(
            pd.DataFrame([[i] for i in range(10)], columns=['i']),
            [BQScalarType.INTEGER])

        table1 = TypedDataFrame(
            pd.DataFrame(
                [[1, 2, 3],
                 [2, 3, 4],
                 [3, 4, 5]],
                columns=['a', 'b', 'c']),
            [BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER])

        table2 = TypedDataFrame(
            pd.DataFrame(
                [[1, 6, 0],
                 [2, 7, 1],
                 [2, 8, 2]],
                columns=['a', 'd', 'e']),
            [BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER])

        table3 = TypedDataFrame(
            pd.DataFrame(
                [[1, 1, 0],
                 [2, 1, 0],
                 [3, 1, 3]],
                columns=['a', 'b', 'c']),
            [BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER])

        counts = TypedDataFrame(
            pd.DataFrame(
                [
                    [1],
                    [1],
                    [2],
                    [None]
                ],
                columns=['i']),
            [BQScalarType.INTEGER])

        timetable = TypedDataFrame(
            pd.DataFrame(
                [[datetime.datetime(2001, 2, 3, 4, 5, 6, 789)]],
                columns=['t']),
            [BQScalarType.DATETIME])

        self.datasets = {
            'my_project': {
                'my_dataset': {
                    'ten_rows': ten_rows,
                    'table1': table1,
                    'table2': table2,
                    'table3': table3,
                    'counts': counts,
                    'timetable': timetable,
                 }
            }
        }

    @data(
        ('1 + 2', 3),
        ('"foo" + \'bar\'', 'foobar'),
        ('4 - 3', 1),
        ('2 * 3', 6),
        ('2 * 3 + 1', 7),
        ('2 * (3 + 1)', 8),
        ('6 / 2 + 1', 4),
        ('- 2', -2),
        ('- (3 + 4)', -7),
        ('2 = 2', True),
        ('2 != 2', False),
        ('3 < 4', True),
        ('CONCAT("A", "B")', 'AB'),
        ('MOD(20, 3)', 2),
        ('(1 = 1) AND (3 = 3)', True),
        ('(1 = 2) AND (3 = 3)', False),
        ('(1 = 1) AND (2 = 3)', False),
        ('(1 = 2) AND (2 = 3)', False),
        ('(1 = 1) OR (3 = 3)', True),
        ('(1 = 2) OR (3 = 3)', True),
        ('(1 = 1) OR (2 = 3)', True),
        ('(1 = 2) OR (2 = 3)', False),
        ('NOT (2=2)', False),
        ('NOT (2=3)', True),
        ('5 IS NOT NULL', True),
        ('5 IS NULL', False),
        ('NULL IS NOT NULL', False),
        ('NULL IS NULL', True),
        ('5 IN (1, 2, 3)', False),
        ('5 IN (1, 2, 5)', True),
        ('5 NOT IN (1, 2, 3)', True),
        ('5 NOT IN (1, 2, 5)', False),
        ('IF(2=3, "yes", "no")', "no"),
        ('IF(2=2, "yes", "no")', "yes"),
        ('TRUE', True),
        ('FALSE', False),
        ('CASE WHEN 1 > 0 THEN 1 ELSE 0 END', 1),
        ('CASE WHEN 1 < 0 THEN 1 ELSE 0 END', 0),
        ('CASE WHEN FALSE THEN 0 WHEN TRUE THEN 1 ELSE 2 END', 1),
        ('CAST(123 AS STRING)', '123'),
        ('CAST(2+2 AS STRING)', '4'),
        ('ARRAY<INT64>[1,2,3]', (1, 2, 3)),
        ('[1,2,3]', (1, 2, 3)),
        ('ARRAY<INT64>[]', ()),
    )
    @unpack
    def test_scalar_expressions(self, expression, expected_result):
        # type: (str, Any) -> None
        tokens = query.tokenize(expression)

        ast, leftover = query.apply_rule(expression_rule, tokens)
        self.assertFalse(leftover, 'leftover {}'.format(leftover))

        assert isinstance(ast, EvaluatableNode)
        typed_series = ast.evaluate(EMPTY_CONTEXT)
        assert isinstance(typed_series, TypedSeries)
        self.assertEqual(typed_series.to_list(), [expected_result])

    @data(*(
        ('select * from `my_project.my_dataset.table1`',
         [[1, 2, 3], [2, 3, 4], [3, 4, 5]]),

        ('select table1.a from `my_project.my_dataset.table1`',
         [[1], [2], [3]]),

        ('select a from `my_project.my_dataset.table1`',
         [[1], [2], [3]]),

        ('select a from `my_project.my_dataset.table1` LIMIT 2',
         [[1], [2]]),

        ('select a from `my_project.my_dataset.table1` LIMIT 1 OFFSET 2',
         [[3]]),

        ('select a from `my_project.my_dataset.table1` ORDER BY a',
         [[1], [2], [3]]),

        ('select a, b from `my_project.my_dataset.table1` ORDER BY a DESC',
         [[3, 4], [2, 3], [1, 2]]),

        ('select a from `my_project.my_dataset.table1` GROUP BY a',
         [[1], [2], [3]]),

        ('select a from `my_project.my_dataset.table1` WHERE b = 3',
         [[2]]),

        ('select count(*) from `my_project.my_dataset.counts`',
         [[4]]),

        ('select count(1) from `my_project.my_dataset.counts`',
         [[4]]),

        ('select count(i) from `my_project.my_dataset.counts`',
         [[3]]),

        ('select 1 + 2',
         [[3]]),

        ('select count(1)',
         [[1]]),

        ('select count(distinct i) from `my_project.my_dataset.counts`',
         [[2]]),

        ('select sum(i) from `my_project.my_dataset.ten_rows`',
         [[45]]),

        ('select i from `my_project.my_dataset.ten_rows` limit 5 offset 2',
         [[2], [3], [4], [5], [6]]),

        ('select sum(i) from `my_project.my_dataset.ten_rows` where mod(i,2)=0',
         [[20]]),

        ('select a, c from `my_project.my_dataset.table1`',
         [[1, 3], [2, 4], [3, 5]]),

        ('select a in (b, c) from `my_project.my_dataset.table3`',
         [[True], [False], [True]]),

        ('select distinct a from table2',
         [[1], [2]]),

        ('select a, max(d), min(e), sum(e), count(*) '
         'from `my_project.my_dataset.table2` group by table2.a',
         [[1, 6, 0, 0, 1],
          [2, 8, 1, 3, 2]]),

        ('select a, max(d), min(e), sum(e), count(*) '
         'from `my_project.my_dataset.table2` group by a',
         [[1, 6, 0, 0, 1],
          [2, 8, 1, 3, 2]]),

        ('select case a when 1 then "one" when 2 then "two" else "three" end from '
         '`my_project.my_dataset.table1`',
         [['one'], ['two'], ['three']]),

        ('select ARRAY<INT64>[], i from ten_rows limit 3',
         [[(), 0], [(), 1], [(), 2]]),

        ('select a, array_agg(d) as ds from table2 group by a',
         [[1, (6, )],
          [2, (7, 8)]]),

        ('select exists (select i from `my_project.my_dataset.ten_rows`)',
         [[True]]),

        ('select exists (select i from `my_project.my_dataset.ten_rows` where i < 0)',
         [[False]]),

        ('select exists(select * from `my_project.my_dataset.table3` '
         'where table1.a = table3.b) from `my_project.my_dataset.table1`',
         [[True], [False], [False]]),

        ('select exists(select * from `my_project.my_dataset.table3` '
         'where table1.a = table3.b) from `my_project.my_dataset.table1` join '
         '`my_project.my_dataset.table3` on table1.a = table3.a',
         [[True], [False], [False]]),

        ) + tuple(
        ('select extract({} from t) from `my_project.my_dataset.timetable`'.format(part),
         [[result]])
        for part, result in (
            ('year', 2001),
            ('month', 2),
            ('day', 3),
            ('dayofweek', 5),
        ))

    )
    @unpack
    def test_scalar_selects(self, sql_query, expected_result):
        result = query.execute_query(sql_query, self.datasets)
        self.assertEqual(result.to_list_of_lists(), expected_result)

    @data(('',
           [[1., 1., 10., 100.]]),
          ('inner',
           [[1., 1., 10., 100.]]),
          ('left',
           [[1., 1., 10., 100.],
            [2., None, 20., None]]),
          ('left outer',
           [[1., 1., 10., 100.],
            [2., None, 20., None]]),
          ('right',
           [[1., 1., 10., 100.],
            [None, 3., None, 300.]]),
          ('right outer',
           [[1., 1., 10., 100.],
            [None, 3., None, 300.]]),
          ('full outer',
           [[1., 1., 10., 100.],
            [2., None, 20., None],
            [None, 3., None, 300.]]),
          )
    @unpack
    def test_join_types(self, join_type, expected_result):
        # Missing join stuff to test.
        # alias with AS; alias without AS
        self.datasets['my_project']['my_dataset']['lefty'] = TypedDataFrame(pd.DataFrame(
            [[1., 10.],
             [2., 20.]],
            columns=['a', 'b']), types=[BQScalarType.FLOAT, BQScalarType.FLOAT])
        self.datasets['my_project']['my_dataset']['righty'] = TypedDataFrame(pd.DataFrame(
            [[1., 100.],
             [3., 300.]],
            columns=['a', 'c']), types=[BQScalarType.FLOAT, BQScalarType.FLOAT])
        sql_query = (
            ('select lefty.a, righty.a, b, c from `my_project.my_dataset.lefty`'
             '{} join `my_project.my_dataset.righty` on lefty.a=righty.a').format(join_type))
        result = query.execute_query(sql_query, self.datasets)
        self.assertEqual(result.to_list_of_lists(), expected_result)

    @data(
        ('on lefty.a=righty.a',
         [[1., 1., 10., 100.],
          [2., None, 20., None],
          [None, 3., None, 300.]]),
        ('using (a)',
         [[1., 1., 10., 100.],
          [2., None, 20., None],
          [None, 3., None, 300.]]),
    )
    @unpack
    def test_join_conditions(self, condition, expected_result):
        self.datasets['my_project']['my_dataset']['lefty'] = TypedDataFrame(pd.DataFrame(
            [[1., 10.],
             [2., 20.]],
            columns=['a', 'b']), types=[BQScalarType.FLOAT, BQScalarType.FLOAT])
        self.datasets['my_project']['my_dataset']['righty'] = TypedDataFrame(pd.DataFrame(
            [[1., 100.],
             [3., 300.]],
            columns=['a', 'c']), types=[BQScalarType.FLOAT, BQScalarType.FLOAT])
        sql_query = (
            ('select lefty.a,righty.a,b,c from `my_project.my_dataset.lefty`'
             'full outer join `my_project.my_dataset.righty` {}').format(condition))
        result = query.execute_query(sql_query, self.datasets)
        self.assertEqual(result.to_list_of_lists(), expected_result)

    def test_leftover_error(self):
        '''A nonsensical query, to trigger "Could not fully parse query"'''

        sql_query = 'a b c'
        expected_error = (
            "Could not fully parse query: leftover tokens \\['a', 'b', 'c'\\]\n"
            "simplified query 'a b c'\n"
            "raw query 'a b c'")
        with self.assertRaisesRegexp(RuntimeError, expected_error):
            query.execute_query(sql_query, self.datasets)

    def test_query_error(self):
        '''Issue a query with insufficient context, to trigger general Exception catch'''

        sql_query = 'SELECT * FROM SomeTable'
        expected_error = (r"Attempt to look up path \('SomeTable',\) "
                          r"with no projects/datasets/tables given")
        with self.assertRaisesRegexp(ValueError, expected_error):
            query.execute_query(sql_query, {})

    def test_simplify_query(self):
        '''Test cleaning up comments and extra whitespace'''

        sql_query = 'SELECT     *\n FROM --Get everything from the table\nSomeTable'
        simplified_query = 'SELECT * FROM SomeTable'
        self.assertEqual(query._simplify_query(sql_query), simplified_query)

    def test_group_by_error(self):
        '''Test that selecting a varying non-group-by-key raises an error'''

        sql_query = 'SELECT d FROM `my_project.my_dataset.table2` GROUP BY a'

        with self.assertRaisesRegexp(ValueError, "not aggregated or grouped by"):
            query.execute_query(sql_query, self.datasets)


if __name__ == '__main__':
    unittest.main()
