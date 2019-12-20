# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
import unittest
from typing import List  # noqa: F401

import pandas as pd
from ddt import data, ddt, unpack

from purplequery.bq_types import BQScalarType, TypedDataFrame
from purplequery.query_helper import apply_rule
from purplequery.statement_grammar import statement as statement_rule
from purplequery.statements import Statement
from purplequery.storage import DatasetTableContext
from purplequery.tokenizer import tokenize


@ddt
class StatementTest(unittest.TestCase):

    @data(
        dict(statement='CREATE TABLE project.dataset.table (a int64, b string);',
             already_exists=False),
        dict(statement='CREATE TABLE IF NOT EXISTS project.dataset.table (a int64, b string);',
             already_exists=False),
        dict(statement='CREATE OR REPLACE TABLE project.dataset.table (a int64, b string);',
             already_exists=True),
        dict(statement='CREATE OR REPLACE TABLE project.dataset.table (a int64, b string);',
             already_exists=False),
    )
    @unpack
    def test_create_table(self, statement, already_exists):
        # type: (str, bool) -> None
        node, leftover = apply_rule(statement_rule, tokenize(statement))
        self.assertFalse(leftover)
        table_context = DatasetTableContext({'project': {'dataset': {}}})
        original_table = TypedDataFrame(pd.DataFrame([], columns=['x', 'y', 'z']),
                                        [BQScalarType.STRING, BQScalarType.INTEGER,
                                         BQScalarType.BOOLEAN])
        if already_exists:
            table_context.set(('project', 'dataset', 'table'), original_table)
        assert isinstance(node, Statement)
        result = node.execute(table_context)
        self.assertEqual(result.path, ('project', 'dataset', 'table'))
        table, unused_name = table_context.lookup(result.path)
        self.assertEqual(list(table.dataframe.columns), ['a', 'b'])
        self.assertEqual(table.types, [BQScalarType.INTEGER, BQScalarType.STRING])

    def test_create_table_already_exists(self):
        # type: () -> None
        node, leftover = apply_rule(statement_rule, tokenize(
            'CREATE TABLE project.dataset.table (a int64, b string);'))
        self.assertFalse(leftover)
        table_context = DatasetTableContext({'project': {'dataset': {}}})
        original_table = TypedDataFrame(pd.DataFrame([], columns=['x', 'y', 'z']),
                                        [BQScalarType.STRING, BQScalarType.INTEGER,
                                         BQScalarType.BOOLEAN])
        table_context.set(('project', 'dataset', 'table'), original_table)
        assert isinstance(node, Statement)
        with self.assertRaisesRegexp(ValueError, 'Already Exists'):
            node.execute(table_context)
            return

    def test_create_table_if_not_exists_and_it_does(self):
        # type: () -> None
        node, leftover = apply_rule(statement_rule, tokenize(
            'CREATE TABLE IF NOT EXISTS project.dataset.table (a int64, b string);'))
        self.assertFalse(leftover)
        table_context = DatasetTableContext({'project': {'dataset': {}}})
        original_table = TypedDataFrame(pd.DataFrame([], columns=['x', 'y', 'z']),
                                        [BQScalarType.STRING, BQScalarType.INTEGER,
                                         BQScalarType.BOOLEAN])
        table_context.set(('project', 'dataset', 'table'), original_table)
        assert isinstance(node, Statement)
        result = node.execute(table_context)
        self.assertEqual(result.path, ('project', 'dataset', 'table'))
        table, unused_name = table_context.lookup(result.path)
        self.assertIs(table, original_table)

    @data(
        dict(statement=('CREATE TABLE project.dataset.table (a int64, b string) '
                        'as (select 1 as x, "hi" as y);'),
             columns=['a', 'b']),
        dict(statement='CREATE TABLE project.dataset.table as (select 1 as x, "hi" as y);',
             columns=['x', 'y']),
    )
    @unpack
    def test_create_table_with_select(self, statement, columns):
        # type: (str, List[str]) -> None
        node, leftover = apply_rule(statement_rule, tokenize(statement))
        self.assertFalse(leftover)
        table_context = DatasetTableContext({'project': {'dataset': {}}})
        assert isinstance(node, Statement)
        result = node.execute(table_context)
        self.assertEqual(result.path, ('project', 'dataset', 'table'))
        table, unused_name = table_context.lookup(result.path)
        self.assertEqual(list(table.dataframe.columns), columns)
        self.assertEqual(table.types, [BQScalarType.INTEGER, BQScalarType.STRING])

    @data(
        dict(query=('create table project.dataset.table (x string)'
                    ' as (select 1 as x)'),
             error='Cannot implicitly coerce the given types'),
        dict(query=('create table project.dataset.table (x int64)'
                    ' as (select 1.5 as x)'),
             error='data of more general type'),
    )
    @unpack
    def test_create_table_with_select_mismatched_types(self, query, error):
        # type: (str, str) -> None
        node, leftover = apply_rule(statement_rule, tokenize(query))
        self.assertFalse(leftover)
        table_context = DatasetTableContext({'project': {'dataset': {}}})
        assert isinstance(node, Statement)
        with self.assertRaisesRegexp(ValueError, error):
            node.execute(table_context)


if __name__ == '__main__':
    unittest.main()
