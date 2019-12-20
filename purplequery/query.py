# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''Run queries against the BigQuery fake implementation.'''

import re

from .bq_abstract_syntax_tree import DatasetType, Result  # noqa: F401
from .dataframe_node import QueryExpression
from .query_helper import apply_rule
from .statement_grammar import bigquery_statement
from .statements import Statement
from .storage import DatasetTableContext
from .tokenizer import remove_comments, tokenize


def _simplify_query(query):
    # type: (str) -> str
    '''Remove comments and extra whitespace from the query string

    Args:
        query: The query as one string
    Returns:
        A new query string without comments and extra whitespace
    '''
    return re.sub(r'[\n\s]+', ' ', remove_comments(query))


def execute_query(query, datasets):
    # type: (str, DatasetType) -> Result
    '''Entrypoint method to run a query against the specified database.

    Args:
        query: The SQL query as a string
        datasets: A representation of all the data in this universe in the
            DatasetType format (see bq_abstract_syntax_tree.py)
    Returns:
        A Result object containing the results of the SQL query on the given data
    '''
    try:
        tokens = tokenize(query)
        tree, leftover = apply_rule(bigquery_statement, tokens)
        if leftover:
            raise RuntimeError('Could not fully parse query: leftover tokens {!r}'.format(leftover))
        if not isinstance(tree, tuple) or len(tree) != 2:
            raise RuntimeError('Parsing expression did not return appropriate data type: {!r}'
                               .format(tree))
        node, unused_optional_semicolon = tree
        table_context = DatasetTableContext(datasets)
        if isinstance(node, (QueryExpression, Statement)):
            return node.execute(table_context)
        raise RuntimeError('Parsing expression did not return appropriate data type: {!r}'
                           .format(node))
    except Exception as e:
        first = e.args[0] if len(e.args) > 0 else ''
        rest = e.args[1:] if len(e.args) > 1 else tuple()
        e.args = (first + "\nsimplified query {!r}\nraw query {!r}".format(
            _simplify_query(query), query),) + rest
        raise
