# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Implementation of BigQuery Statements."""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple  # noqa: F401

import pandas as pd

from .bq_abstract_syntax_tree import TableContext  # noqa: F401
from .bq_abstract_syntax_tree import AbstractSyntaxTreeNode, Result, _EmptyNode
from .bq_types import BQType, TypedDataFrame, implicitly_coerce  # noqa: F401
from .dataframe_node import QueryExpression  # noqa: F401


class Statement(AbstractSyntaxTreeNode):
    '''BigQuery statement that mutates the environment.'''

    @abstractmethod
    def execute(self, table_context):
        # type: (TableContext) -> Result
        '''Executes the statement to modify the environment.

        Args:
            table_context: the currently existing datasets in the environment.
        Returns:
            The statement type: see
            https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.statement_type
        '''


class CreateTable(Statement):
    '''A CREATE TABLE statement.'''

    def __init__(self, create_type, path, schema, options, query_expression):
        # type: (str, Tuple[str, ...], List[Tuple[str, BQType]], _EmptyNode, QueryExpression) -> None  # noqa: E501
        self.create_type = create_type
        self.path = path
        self.schema = schema
        self.options = options
        self.query_expression = query_expression

    def execute(self, table_context):
        # type: (TableContext) -> Result
        '''See parent class for docstring.'''
        statement_type = ('CREATE_TABLE'
                          if isinstance(self.query_expression, _EmptyNode)
                          else 'CREATE_TABLE_AS_SELECT')
        result = Result(statement_type, path=self.path)
        try:
            table_context.lookup(self.path)
            already_exists = True
        except KeyError:
            already_exists = False
        if already_exists and self.create_type == 'CREATE_TABLE':
            raise ValueError('Already Exists')
        if already_exists and self.create_type == 'CREATE_TABLE_IF_NOT_EXISTS':
            return result

        table = None
        if not isinstance(self.query_expression, _EmptyNode):
            table, _ = self.query_expression.get_dataframe(table_context)

        if not isinstance(self.schema, _EmptyNode):
            columns = [column for column, type_ in self.schema]
            types = [type_ for column, type_ in self.schema]
            if table is None:
                table = TypedDataFrame(pd.DataFrame([], columns=columns), types)
            else:
                table.dataframe.columns = columns
                for column, query_type, explicit_type in zip(columns, table.types, types):
                    if explicit_type != implicitly_coerce(query_type, explicit_type):
                        raise ValueError(
                            ("Creating table {}; column has schema type {} "
                             "but data of more general type {}").format(
                                 '.'.join(self.path), column, explicit_type, query_type))
        table_context.set(self.path, table)
        return result


class CreateView(Statement):
    '''A CREATE VIEW Statement'''

    def __init__(self, create_type, path, options, query_expression):
        # type: (str, List[str], _EmptyNode, QueryExpression) -> None
        self.create_type = create_type
        self.path = path
        self.options = options
        self.query_expression = query_expression

    def execute(self, table_context):
        # type: (TableContext) -> Result
        '''See parent class for docstring.'''
        statement_type = 'CREATE_VIEW'
        # TODO: Actually implement CREATE VIEW
        return Result(statement_type, path=self.path)
