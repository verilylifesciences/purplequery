import unittest

from ddt import data, ddt, unpack

from purplequery.dataframe_node import QueryExpression
from purplequery.query_helper import apply_rule
from purplequery.statement_grammar import statement as statement_rule
from purplequery.statement_grammar import bigquery_statement
from purplequery.statements import CreateTable, CreateView
from purplequery.tokenizer import tokenize


@ddt
class StatementGrammarTest(unittest.TestCase):

    @data(
        dict(statement='select 1', type_=QueryExpression),
        dict(statement='select 1;', type_=QueryExpression),
        dict(statement='create table foo.bar.baz', type_=CreateTable),
        dict(statement='create table foo.bar.baz;', type_=CreateTable),
    )
    @unpack
    def test_bigquery_statement(self, statement, type_):
        # type: (str, type) -> None
        tree, leftover = apply_rule(bigquery_statement, tokenize(statement))
        self.assertFalse(leftover)
        assert isinstance(tree, tuple)
        node, unused_semicolon = tree
        self.assertIsInstance(node, type_)

    @data(
        *[{'statement': ' '.join((create, identifier, options, 'as', '(select 1)'))}
          for create in ('create view if not exists', 'create view', 'create or replace view')
          for identifier in ('foo', 'foo.bar', 'foo.bar.baz')
          for options in ('options()', 'options(a=b)', 'options(a=b, c=2)')]
    )
    @unpack
    def test_create_view_grammar(self, statement):
        # type: (str) -> None
        node, leftover = apply_rule(statement_rule, tokenize(statement))
        self.assertFalse(leftover)
        self.assertIsInstance(node, CreateView)

    @data(
        *[{'statement': ' '.join((create, identifier, options, maybe_query))}
          for create in ('create table if not exists', 'create table', 'create or replace table')
          for identifier in ('foo', 'foo.bar', 'foo.bar.baz')
          for options in ('options()', 'options(a=b)', 'options(a=b, c=2)')
          for maybe_query in ('as (select 1)', '')]
    )
    @unpack
    def test_create_table_grammar(self, statement):
        # type: (str) -> None
        node, leftover = apply_rule(statement_rule, tokenize(statement))
        self.assertFalse(leftover)
        self.assertIsInstance(node, CreateTable)


if __name__ == '__main__':
    unittest.main()
