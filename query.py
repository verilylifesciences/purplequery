# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''Run queries against the BigQuery fake implementation.'''

import re

from bq_abstract_syntax_tree import DatasetTableContext, DatasetType  # noqa: F401
from bq_types import TypedDataFrame  # noqa: F401
from dataframe_node import QueryExpression
from grammar import query_expression
from query_helper import apply_rule
from tokenizer import tokenize


def _simplify_query(query):
    # type: (str) -> str
    '''Remove comments and extra whitespace from the query string

    Args:
        query: The query as one string
    Returns:
        A new query string without comments and extra whitespace
    '''
    prev, cur = None, query
    # There's a bug in re.sub where it doesn't work for adjacent matches
    while prev != cur:
        prev = cur
        cur = re.sub(r'--.*\n', '', prev)
    return re.sub(r'[\n\s]+', ' ', cur)


def execute_query(query, datasets):
    # type: (str, DatasetType) -> TypedDataFrame
    '''Entrypoint method to run a query against the specified database.

    Args:
        query: The SQL query as a string
        datasets: A representation of all the data in this universe in the
        DatasetType format (see bq_abstract_syntax_tree.py)
    Returns:
        A table (TypedDataFrame) containing the results of the SQL query on
        the given data
    '''
    try:
        tokens = tokenize(query)
        tree, leftover = apply_rule(query_expression, tokens)
        if leftover:
            raise RuntimeError('Could not fully parse query: leftover tokens {!r}'.format(leftover))
        if not isinstance(tree, QueryExpression):
            raise RuntimeError('Parsing expression did not return appropriate data type: {!r}'
                               .format(tree))
        typed_dataframe, _ = tree.get_dataframe(DatasetTableContext(datasets))
    except Exception as e:
        first = e.args[0] if len(e.args) > 0 else ''
        rest = e.args[1:] if len(e.args) > 1 else tuple()
        e.args = (first + "\nsimplified query {!r}\nraw query {!r}".format(
            _simplify_query(query), query),) + rest
        raise
    return typed_dataframe
