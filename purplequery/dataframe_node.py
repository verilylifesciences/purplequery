# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''All subclasses of DataframeNode'''

import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union  # noqa: F401

import pandas as pd

from six.moves import reduce

from .bq_abstract_syntax_tree import (EMPTY_CONTEXT, EMPTY_NODE,  # noqa: F401
                                      AbstractSyntaxTreeNode, DataframeNode, DatasetType,
                                      EvaluatableNode, EvaluationContext, Field,
                                      MarkerSyntaxTreeNode, TableContext, _EmptyNode)
from .bq_types import (BQArray, BQStructType, BQType, TypedDataFrame, TypedSeries,  # noqa: F401
                       implicitly_coerce)
from .evaluatable_node import Array, Selector, StarSelector, Value  # noqa: F401
from .join import DataSource  # noqa: F401

DEFAULT_TABLE_NAME = None

_OrderByType = List[Tuple[Field, str]]
_LimitType = Tuple[EvaluatableNode, EvaluatableNode]


class _WithTableContext(TableContext):
    '''A TableContext augmented by a WITH clause.'''

    def __init__(self, name, table, parent_context):
        # type: (str, TypedDataFrame, TableContext) -> None
        self.name = name
        self.table = table
        self.parent_context = parent_context

    def lookup(self, path):
        # type: (Tuple[str, ...]) -> Tuple[TypedDataFrame, Optional[str]]
        '''Look up a path to a table in this context.'''

        if len(path) == 1 and path[0] == self.name:
            return self.table, self.name
        return self.parent_context.lookup(path)


class QueryExpression(DataframeNode):
    '''Highest level definition of a query.

    https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#sql-syntax
    (see query_expr)
    '''

    def __init__(self,
                 with_clauses,  # type: Union[_EmptyNode, List[Tuple[str, DataframeNode]]]
                 base_query,  # type: DataframeNode
                 order_by,  # type: Union[_EmptyNode, _OrderByType]
                 limit,  # type: Union[_EmptyNode, _LimitType]
                 ):
        # type: (...) -> None
        '''Set up QueryExpression node.

        Args:
            with_clauses: Optional WITH expression
            base_query: Main part of query
            order_by: Expression by which to order results
            limit: Number of rows to return, possibly with an offset
        '''

        self.with_clauses = with_clauses
        self.base_query = base_query
        self.order_by = order_by
        self.limit = limit

    def _order_by(self, order_by, typed_dataframe, table_name, table_context):
        # type: (_OrderByType, TypedDataFrame, Optional[str], TableContext) -> TypedDataFrame
        '''If ORDER BY is specified, sort the data by the given column(s)
        in the given direction(s).

        Args:
            typed_dataframe: The currently resolved query as a TypedDataFrame
            table_name: Resolved name of current typed_dataframe
            table_context: A representation of the state of available tables
        Returns:
            A new TypedDataFrame that is ordered by the given criteria
        '''
        context = EvaluationContext(table_context)
        context.add_table_from_dataframe(typed_dataframe, table_name, EMPTY_NODE)

        # order_by is a list of (field, direction) tuples to sort by
        fields = []
        directions = []  # ascending = True, descending = False
        for field, direction in order_by:
            if isinstance(field, Field):
                path = '.'.join(context.get_canonical_path(field.path))
                fields.append(path)
            elif isinstance(field, Value):
                if not isinstance(field.value, int):
                    raise ValueError('Attempt to order by a literal non-integer constant {}'
                                     .format(field.value))
                index = field.value - 1  # order by 1 means the first field, i.e. index 0
                fields.append(context.table.dataframe.columns[index])
            else:
                raise ValueError('Invalid field specification {}'.format(field))

            if direction == 'DESC':
                directions.append(False)
            else:
                # Default sort order in Standard SQL is ASC
                directions.append(True)
        return TypedDataFrame(
            context.table.dataframe.sort_values(fields, ascending=directions),
            context.table.types)

    def _limit(self, limit, typed_dataframe):
        # type: (_LimitType, TypedDataFrame) -> TypedDataFrame
        '''If limit is specified, only return that many rows.
        If offset is specified, start at that row number, not the first row.

        Args:
            typed_dataframe: The currently resolved query as a TypedDataFrame
        Returns:
            A new TypedDataFrame that conforms to the given limit and offset
        '''
        limit_expression, offset_expression = limit

        # Use empty context because the limit is a constant
        limit_value = limit_expression.evaluate(EMPTY_CONTEXT)
        if not isinstance(limit_value, TypedSeries):
            raise ValueError("invalid limit expression {}".format(limit_expression))
        limit, = limit_value.series
        if offset_expression is not EMPTY_NODE:
            # Use empty context because the offset is also a constant
            offset_value = offset_expression.evaluate(EMPTY_CONTEXT)
            if not isinstance(offset_value, TypedSeries):
                raise ValueError("invalid offset expression {}".format(offset_expression))
            offset, = offset_value.series
        else:
            offset = 0
        return TypedDataFrame(
            typed_dataframe.dataframe[offset:limit + offset],
            typed_dataframe.types)

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''See parent, DataframeNode'''
        if not isinstance(self.with_clauses, _EmptyNode):
            name_list = [name for name, _ in self.with_clauses]
            if len(name_list) > len(set(name_list)):
                raise ValueError("Duplicate names in WITH clauses are not allowed: {}"
                                 .format(name_list))
            for name, dataframe_node in self.with_clauses:
                table_context = _WithTableContext(name,
                                                  dataframe_node.get_dataframe(table_context)[0],
                                                  table_context)

        typed_dataframe, table_name = self.base_query.get_dataframe(table_context, outer_context)

        if not isinstance(self.order_by, _EmptyNode):
            typed_dataframe = self._order_by(
                self.order_by, typed_dataframe, table_name, table_context)

        if not isinstance(self.limit, _EmptyNode):
            typed_dataframe = self._limit(self.limit, typed_dataframe)

        return typed_dataframe, DEFAULT_TABLE_NAME


class SetOperation(DataframeNode):
    '''Represents a set operation between two other query expressions - UNION, INTERSECT, etc.'''

    def __init__(self, left_query, set_operator, right_query):
        # type: (DataframeNode, str, DataframeNode) -> None
        self.left_query = left_query
        self.set_operator = set_operator
        self.right_query = right_query

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''See parent, DataframeNode'''
        left_dataframe, unused_left_name = self.left_query.get_dataframe(
            table_context, outer_context)
        right_dataframe, unused_right_name = self.right_query.get_dataframe(
            table_context, outer_context)
        num_left_columns = len(left_dataframe.types)
        num_right_columns = len(right_dataframe.types)
        if num_left_columns != num_right_columns:
            raise ValueError("Queries in UNION ALL have mismatched column count: {} vs {}"
                             .format(num_left_columns, num_right_columns))
        combined_types = [implicitly_coerce(left_type, right_type)
                          for left_type, right_type in zip(left_dataframe.types,
                                                           right_dataframe.types)]
        if self.set_operator == 'UNION_ALL':
            return TypedDataFrame(
                pd.concat([left_dataframe.dataframe, right_dataframe.dataframe]),
                combined_types), DEFAULT_TABLE_NAME
        else:
            raise NotImplementedError("set operation {} not implemented".format(self.set_operator))


def _evaluate_fields_as_dataframe(fields, context):
    # type: (Sequence[EvaluatableNode], EvaluationContext) -> TypedDataFrame
    '''Evaluates a list of expressions and constructs a TypedDataFrame from the result.

    Args:
        fields: A list of expressions (evaluatable abstract syntax tree nodes)
        context: The context to evaluate the expressions
    Returns:
        A TypedDataFrame consisting of the results of the evaluation.
    '''
    # Evaluates each of the given fields to get a list of tables and/or
    # single columns
    evaluated_fields = [field.evaluate(context) for field in fields]

    # Creates one large table out of each of the evaluated field
    # tables/columns
    types = reduce(operator.add,
                   [field.types for field in evaluated_fields], [])  # type: List[BQType]
    combined_evaluated_data = (
            pd.concat([field.dataframe for field in evaluated_fields], axis=1)
            if evaluated_fields else pd.DataFrame([]))
    return TypedDataFrame(combined_evaluated_data, types)


class Select(MarkerSyntaxTreeNode, DataframeNode):
    '''SELECT query to retrieve rows from a table(s).

    https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#select-list
    '''

    def __init__(self, modifier,  # type: AbstractSyntaxTreeNode
                 fields,  # type: Sequence[Union[Selector, StarSelector]]
                 from_,  # type: Union[_EmptyNode, DataSource]
                 where,  # type: Union[_EmptyNode, EvaluatableNode]
                 group_by,  # type: Union[_EmptyNode, List[Union[Value, Field]]]
                 having  # type: Union[_EmptyNode, EvaluatableNode]
                 ):
        # type: (...) -> None
        '''Set up SELECT node.

        Args:
            modifier: Optional ALL or DISTINCT modifier
            fields: Columns to return
            from_: Table/expression from which to retrieve rows
            where: WHERE filter condition, if any
            group_by: GROUP BY grouping condition, if any
            having: HAVING filter condition, if any
        '''
        self.modifier = modifier
        self.fields = fields
        for i, field in enumerate(self.fields):
            field.position = i + 1  # position is 1-up, i.e the first selector is position #1.
        self.from_ = from_
        self.where = where
        if isinstance(group_by, _EmptyNode):
            self.group_by = group_by  # type: Union[_EmptyNode, List[Field]]
        else:
            self.group_by = []
            for grouper in group_by:
                if isinstance(grouper, Value):
                    if not isinstance(grouper.value, int):
                        raise ValueError('Attempt to group by a literal non-integer constant {}'
                                         .format(grouper.value))
                    # GROUP BY 3 means group by the third field in the select, the field at index 2,
                    # i.e. we have to subtract one from the user-specified value to get the index.
                    # We construct a one-element field path just as if they'd specified the name
                    # of the corresponding field.
                    grouper_path = (self.fields[grouper.value - 1].name(),)
                    self.group_by.append(Field(grouper_path))
                else:
                    self.group_by.append(grouper)
        self.having = having

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''Scope the given datasets by the criteria specified in the
        instance's fields.

        Args:
            table_context: All the tables in the database
            outer_context: The context of the outer query, if this Select is a subquery;
                otherwise None
        Returns:
            Tuple of the resulting table (TypedDataFrame) and a name for
            this table
        '''

        if isinstance(self.from_, _EmptyNode):
            context = EvaluationContext(table_context)
        else:
            context = self.from_.create_context(table_context)

        if outer_context:
            context.add_subcontext(outer_context)

        context.selector_names = [
                selector.name() for selector in self.fields if isinstance(selector, Selector)]

        if not isinstance(self.where, _EmptyNode):
            # Filter table by WHERE condition
            rows_to_keep = self.where.evaluate(context)
            if not isinstance(rows_to_keep, TypedSeries):
                raise ValueError("Invalid WHERE expression {}".format(rows_to_keep))
            context.table = TypedDataFrame(
                context.table.dataframe.loc[rows_to_keep.series],
                context.table.types)

        if not isinstance(self.group_by, _EmptyNode):
            fields_for_evaluation = context.do_group_by(
                self.fields, self.group_by)  # type: Sequence[EvaluatableNode]
        else:
            fields_for_evaluation = self.fields
        result = _evaluate_fields_as_dataframe(fields_for_evaluation, context)

        if not isinstance(self.having, _EmptyNode):
            having_context = EvaluationContext(table_context)
            having_context.add_table_from_dataframe(result, None, EMPTY_NODE)
            having_context.add_subcontext(context)
            having_context.group_by_paths = context.group_by_paths
            having = self.having.mark_grouped_by(context.group_by_paths, having_context)
            rows_to_keep = having.evaluate(having_context)
            if not isinstance(rows_to_keep, TypedSeries):
                raise ValueError("Invalid HAVING expression {}".format(rows_to_keep))
            result = TypedDataFrame(result.dataframe.loc[rows_to_keep.series], result.types)

        if self.modifier == 'DISTINCT':
            result = TypedDataFrame(result.dataframe.drop_duplicates(), result.types)

        return result, DEFAULT_TABLE_NAME


class TableReference(DataframeNode):
    '''A table reference specified as Project.Dataset.Table (or possibly
    Dataset.Table or just Table if there is only one project and/or dataset).
    '''

    def __init__(self, path):
        # type: (Tuple[str, ...]) -> None

        # If the table reference is specified with backticks, it will be parsed
        # as one element, so we need to split into parts here.
        if len(path) == 1:
            split_path = path[0].split('.')  # type: List[str]
            path = tuple(split_path)
        self.path = path  # type: Tuple[str, ...]

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''See parent, DataframeNode'''
        del outer_context  # Unused
        return table_context.lookup(self.path)


class Unnest(DataframeNode, MarkerSyntaxTreeNode):
    '''An expression unnesting an array into a column of data.'''

    def __init__(self, array_node):
        # type: (Array) -> None

        self.array_node = array_node

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''See parent, DataframeNode'''
        del outer_context  # Unused
        context = EvaluationContext(table_context)
        result = self.array_node.evaluate(context)
        if isinstance(result, TypedDataFrame):
            raise ValueError('UNNEST({}) did not result in one column'.format(self.array_node))
        result_type, = result.types

        if not isinstance(result_type, BQArray):
            raise ValueError("UNNESTing a non-array-typed value: {}".format(result_type))

        contained_type = result_type.type_

        if len(result.series) != 1:
            raise ValueError('UNNEST({}) did not result in one row'.format(self.array_node))
        result_array, = result.to_list()
        if not isinstance(result_array, tuple):
            raise ValueError("UNNEST({}) resulted in {!r} rather than an array"
                             .format(self.array_node, result_array))
        if isinstance(contained_type, BQStructType):
            result_dataframe = TypedDataFrame(
                pd.DataFrame([[cell for cell in row] for row in result_array],
                             columns=[field if field else '' for field in contained_type.fields]),
                contained_type.types)
        else:
            result_dataframe = TypedDataFrame(
                pd.DataFrame([[cell] for cell in result_array], columns=['']), [contained_type])
        return result_dataframe, None
