# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''Logic implementing JOINs between tables.

DataSource is an abstract syntax tree representing the data pulled into a select statement,
consisting of one or more tables JOINed together.  The create_context method of this class
implements the logic to join the tables together according to specified conditions.
'''
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple,  # noqa: F401
                    Union, cast)

import pandas as pd

from .binary_expression import BinaryExpression
from .bq_abstract_syntax_tree import (EMPTY_NODE, AbstractSyntaxTreeNode,  # noqa: F401
                                      DataframeNode, EvaluatableNode, EvaluationContext, Field,
                                      TableContext, _EmptyNode)
from .bq_types import TypedDataFrame, TypedSeries  # noqa: F401

# column name of ephemeral key added to implement a cross join with pandas merge.
_CROSS_KEY = '__cross_key__'


FromItemType = Tuple[DataframeNode, Union[_EmptyNode, str]]
ConditionsType = Union[_EmptyNode,  # JOIN with no condition
                       Tuple[str, ...],  # JOIN USING (strings)
                       EvaluatableNode,  # JOIN ON (condition)
                       ]

Join = NamedTuple('Join', [('join_type', Union[str, _EmptyNode]),
                           ('join_with_alias', FromItemType),
                           ('join_conditions', ConditionsType)])


def _extract_simple_comparison(node):
    # type: (EvaluatableNode) -> Optional[List[Tuple[Field, Field]]]
    """Extracts a list of pairs of fields if node looks like a = b AND c = d ....

    Args:
        node: Some expression Abstract syntax tree node.
    Returns:
        node represents some evaluatable expression.  If it's just a simple conjunction of equality
        comparisons of fields, then return a list of pairs of those fields.  Otherwise, return None.
    """
    if isinstance(node, BinaryExpression):
        left, right = node.children
        if node.operator_info.operator == 'AND':
            left_comparisons = _extract_simple_comparison(left)
            right_comparisons = _extract_simple_comparison(right)
            if left_comparisons is not None and right_comparisons is not None:
                return left_comparisons + right_comparisons
        elif (node.operator_info.operator == '=' and isinstance(left, Field)
              and isinstance(right, Field)):
            return [(left, right)]
    return None


def _cross_join(left_table, right_table):
    # type: (TypedDataFrame, TypedDataFrame) -> TypedDataFrame
    '''Returns the cross join of the two tables.

    A CROSS join takes the cartesian product of the left and right, i.e. all possible pairs
    of a row from the left and a row from the right.  No conditions are necessary to perform
    this join, and also this join type isn't directly supported by the pandas merge function,
    so we have special handling for it here.  We perform the cross join by adding a constant
    column to both the left and right table and joining on it, which gives the desired result.

    Args:
        left_table: A TypedDataFrame
        right_table: A TypedDataFrame

    Returns:
        The cross-joined table.
    '''
    left_table.dataframe[_CROSS_KEY] = 0
    right_table.dataframe[_CROSS_KEY] = 0
    result = TypedDataFrame(
            left_table.dataframe.merge(
                    right_table.dataframe,
                    # the join type ('how' in pandas) doesn't matter, because
                    # the difference between the join types lies in what to do with rows that
                    # don't match, and here all rows match, so all the join types behave the same.
                    how='inner',
                    on=_CROSS_KEY).drop(_CROSS_KEY, axis=1),
            left_table.types + right_table.types)
    left_table.dataframe.drop(_CROSS_KEY, inplace=True, axis=1)
    right_table.dataframe.drop(_CROSS_KEY, inplace=True, axis=1)
    return result


def _get_common_columns(left_table, right_table):
    # type: (TypedDataFrame, TypedDataFrame) -> Tuple[List[str], List[str]]
    '''Returns commons column to the left and right table.

    When columns are joined to a context, the column name is prefixed with the table name,
    so we make sure to remove that prefix to identify which columns go together, and then put it
    back so we have lists of columns that match either side.

    Args:
        left_table: A TypedDataFrame
        right_table: A TypedDataFrame
    Returns:
        left columns to join on, right columns to join on.
    '''

    left_table_columns = {
            column.split('.')[-1]: column for column in left_table.dataframe.columns}
    right_table_columns = {
            column.split('.')[-1]: column for column in right_table.dataframe.columns}
    shared_columns = list(set(left_table_columns) & set(right_table_columns))
    left_ons = [left_table_columns[col] for col in shared_columns]
    right_ons = [right_table_columns[col] for col in shared_columns]
    return left_ons, right_ons


def _get_join_using(join_condition, right_table_id, context):
    # type: (Sequence[str], str, EvaluationContext) -> Tuple[List[str], List[str]]
    '''Returns the left and right columns matching a list of shared column ids.

    Args:
        join_condition: A tuple of strings, naming columns that appear in both tables.
        right_table_id: The id of the table on the right.
        context: EvaluationContext in which to interpret the ids.
    Returns:
        The columns in the left and right table corresponding to the desired columns.
    '''
    left_ons, right_ons = [], []  # type: Tuple[List[str], List[str]]
    for element in join_condition:
        element_paths = context.get_all_canonical_paths((element,))
        if len(element_paths) != 2:
            raise ValueError(
                "JOIN USING key must exist in exactly two tables; exists in these: {!r}"
                .format([path[0] for path in element_paths]))
        # Determine which field is in the left table and which in the right
        if element_paths[0][0] == right_table_id:
            left_path, right_path = element_paths[1], element_paths[0]
        elif element_paths[1][0] == right_table_id:
            left_path, right_path = element_paths[0], element_paths[1]
        else:
            raise ValueError(
                ("JOIN USING key must exist in joined table"
                 ", exists only in these: {!r}").format(element_paths))
        left_ons.append('.'.join(left_path))
        right_ons.append('.'.join(right_path))
    return left_ons, right_ons


def _get_join_on_equality_comparisons(
        join_comparisons,  # type: List[Tuple[Field, Field]]
        right_table_id,  # type: str
        context  # type: EvaluationContext
):
    # type: (...) -> Tuple[List[str], List[str]]
    '''Returns the columns to join on given a list of equality comparisons.

    The columns provided are user-specified (e.g. ('a', 'b')), whereas what we need to execute
    pandas.merge are the canonicalized DataFrame column names (which are always prefixed with
    a table name).

    Also, the user may specify something like table1.a = table2.b and table2.c = table1.d, but to
    execute the merge we need all the table1 columns in one list and all the table2 columns in
    another list.  So, for each element of each pair, we figure out which list it goes in.

    Args:
        join_comparisons: A list of pairs of user-specified names of columns that must be equal for
            corresponding rows to be joined.
        right_table_id: The id of the table being joined.
        context: The context in which the names are evaluated.

    Returns:
        left columns to join on, right columns to join on.
    '''
    left_ons, right_ons = [], []  # type: Tuple[List[str], List[str]]
    for join_on_field1, join_on_field2 in join_comparisons:
        join_on_field1_path = context.get_canonical_path(join_on_field1.path)
        join_on_field2_path = context.get_canonical_path(join_on_field2.path)
        # Determine which field is in the left table and which in the right
        if right_table_id == join_on_field1_path[0]:
            left_path, right_path = join_on_field2_path, join_on_field1_path
        elif right_table_id == join_on_field2_path[0]:
            left_path, right_path = join_on_field1_path, join_on_field2_path
        else:
            raise ValueError(
                "Neither field {!r} nor {!r} exists in right table {!r}".format(
                    join_on_field1, join_on_field2, right_table_id))
        left_ons.append('.'.join(left_path))
        right_ons.append('.'.join(right_path))
    return left_ons, right_ons


def _join_on_arbitrary_condition(left_table,  # type: TypedDataFrame
                                 right_table,  # type: TypedDataFrame
                                 join_condition,  # type: EvaluatableNode
                                 context,  # type: EvaluationContext
                                 pandas_join_type  # type: str
                                 ):
    # type: (...) -> TypedDataFrame
    '''Returns the result of joining tables on an arbitrary boolean join condition.

    To perform JOINs we are in general relying on the pandas merge function.  However, it does
    not allow performing joins on arbitrary boolean expressions.  So, we simulate that logic
    at a lower level, by doing a cross join (to get all combinations of rows), applying the boolean
    condition, and then if something other than an inner join is requested, adding back the left
    or right rows as needed.

    Args:
        left_table: A TypedDataFrame
        right_table: A TypedDataFrame
        join_condition: A boolean expression true for pairs of rows to keep
        context: The evaluation context to evaluate the condition in
        pandas_join_type: How to join the tables.

    Returns:
        The joined table.
    '''
    context.table = _cross_join(left_table, right_table)
    rows_to_keep = join_condition.evaluate(context)
    if not isinstance(rows_to_keep, TypedSeries):
        raise RuntimeError("join condition {} evaluated to a table rather than a column"
                           .format(join_condition))
    result_dataframe = context.table.dataframe.loc[rows_to_keep.series]
    if pandas_join_type in ['left', 'outer']:
        result_dataframe = pd.concat(
                [result_dataframe, left_table.dataframe]).drop_duplicates(
                        subset=left_table.dataframe.columns)
    if pandas_join_type in ['right', 'outer']:
        result_dataframe = pd.concat(
                [result_dataframe, right_table.dataframe]).drop_duplicates(
                        subset=right_table.dataframe.columns)
    return TypedDataFrame(result_dataframe, context.table.types)


class DataSource(AbstractSyntaxTreeNode):
    '''Node representing JOIN operations.

    Not a child of EvaluatableNode because this evaluate() has a different input and
    output type than the rest of the evaluate()s.
    '''
    BIGQUERY_TO_PANDAS_JOIN_TYPE = {
        EMPTY_NODE: 'inner',
        'CROSS': 'inner',  # Cross join behaves like inner if join conditions are provided.
        'INNER': 'inner',
        'LEFT': 'left',
        'LEFT_OUTER': 'left',
        'FULL': 'outer',
        'FULL_OUTER': 'outer',
        'RIGHT': 'right',
        'RIGHT_OUTER': 'right',
    }

    def __init__(self,
                 first_from,  # type: FromItemType
                 joins  # type: List[Join]
                 ):
        # type: (...) -> None
        '''Set up JOIN node.

        Args:
            first_from: The first table that was specified after FROM
            joins: The rest of the tables, if any, to join to the first
                This is a list, where each element is a tuple of 3 elements:
                join_type: Key into self.BIGQUERY_TO_PANDAS_JOIN_TYPE
                join_with_alias: Tuple of a TableReference and an optional alias
                join_condition: Tuple of fields to join on, where each is
                    either a string (field name) or tuple of 2 Field nodes
        '''
        self.first_from = first_from
        self.joins = joins

    def _get_joined_table(self,
                          context,  # type: EvaluationContext
                          table,  # type: TypedDataFrame
                          join_type,  # type: Union[str, _EmptyNode]
                          join_with_alias,  # type: FromItemType
                          join_condition  # type: ConditionsType
                          ):
        # type: (...) -> TypedDataFrame
        """Gets the TypedDataFrame after performing a join.

        Args:
            context: EvaluationContext in which to evaluate the join.
            table: The table to join to.
            join_type: If provided, the type of join (e.g. INNER, OUTER, etc.)
            join_with_alias: The table to be joined in, with an optional alias
            join_condition: The specified conditions on the join.

        Returns:
            The TypedDataFrame after the join is done.
        """
        join_table, join_table_id = context.add_table_from_node(*join_with_alias)

        join_type = join_type.upper() if isinstance(join_type, str) else join_type

        pandas_join_type = self.BIGQUERY_TO_PANDAS_JOIN_TYPE.get(join_type)
        if pandas_join_type is None:
            raise NotImplementedError("Join type {} is not supported".format(join_type))

        # Now we execute the JOIN operation by determining the type of join condition -- the
        # user-specified condition of which rows from `table' are joined with which rows from
        # `join_table` -- and taking the appropriate action.  As much as possible, we want to use
        # the pandas DataFrame.merge method, which means we the action will be to convert the
        # user-specified join condition into a list of columns on the left and columns on the right
        # (left_ons and right_ons) and call pandas.merge.
        #
        # Two kinds of joins -- unconditional cross joins, and joins on an arbitrary boolean
        # expression -- are not supported by the merge method, and so we have separate logic that
        # directly calculates and returns the merged table.
        if join_type == 'CROSS' and join_condition is EMPTY_NODE:
            return _cross_join(table, join_table)

        # If no specific join condition is given, we join on columns that are common between
        # the two tables.
        elif isinstance(join_condition, _EmptyNode):
            left_ons, right_ons = _get_common_columns(table, join_table)

        # If join USING(list of fields) is specified.
        elif isinstance(join_condition, tuple):
            left_ons, right_ons = _get_join_using(join_condition, join_table_id, context)

        # If join ON is specified
        else:
            join_comparisons = _extract_simple_comparison(join_condition)

            # If join ON (a = b AND c = d AND ...) is specified.
            if join_comparisons is not None:
                left_ons, right_ons = _get_join_on_equality_comparisons(
                        join_comparisons, join_table_id, context)

            # The user can also join ON some arbitrary boolean condition, e.g. JOIN ON (a+b < c).
            else:
                return _join_on_arbitrary_condition(table, join_table, join_condition, context,
                                                    pandas_join_type)

        return TypedDataFrame(table.dataframe.merge(
                join_table.dataframe, how=pandas_join_type, left_on=left_ons, right_on=right_ons),
                              table.types + join_table.types)

    def create_context(self, table_context):
        # type: (TableContext) -> EvaluationContext
        '''Given a representation of the entire database, add the specified
        table(s) to the query's context.
        '''
        context = EvaluationContext(table_context)
        table, _ = context.add_table_from_node(*self.first_from)

        for join_type, join_with_alias, join_condition in self.joins:
            table = self._get_joined_table(
                    context, table, join_type, join_with_alias, join_condition)

        context.table = table
        return context
