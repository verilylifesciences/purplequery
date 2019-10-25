# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''All subclasses of EvaluatableNode'''

import operator
from abc import ABCMeta, abstractmethod
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple,  # noqa: F401
                    Union)

import pandas as pd

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import (EMPTY_NODE, AbstractSyntaxTreeNode,  # noqa: F401
                                     DataframeNode, EvaluatableLeafNode, EvaluatableNode,
                                     EvaluatableNodeThatAggregatesOrGroups,
                                     EvaluatableNodeWithChildren, EvaluationContext, GroupedBy,
                                     MarkerSyntaxTreeNode, _EmptyNode)
from bq_types import (BQScalarType, BQType, TypedDataFrame, TypedSeries,  # noqa: F401
                      implicitly_coerce)
from six.moves import reduce

NoneType = type(None)
LiteralType = Union[NoneType, bool, int, float, str]


class Case(MarkerSyntaxTreeNode, EvaluatableNodeWithChildren):
    '''A CASE expression, similar to an IF.  For example:
    CASE WHEN a > 0 THEN "positive" WHEN a < 0 THEN "negative" ELSE "zero" END
    CASE a WHEN 1 THEN "one" WHEN 2 THEN "two" END
    '''
    def __init__(self, comparand,  # type: AbstractSyntaxTreeNode
                 whens,  # type: List[Tuple[AbstractSyntaxTreeNode, EvaluatableNode]]
                 else_  # type: Union[EvaluatableNode, _EmptyNode]
                 ):
        # type: (...) -> None
        '''Set up Case node

        Args:
            comparand: An expression that should be compared to each of the
                provided WHENs, or an empty node if the whens are expressions
                that evaluate to booleans
            whens: A list of tuples describing each WHEN. The first value is
                either a condition that evaluates to a boolean, or a value to
                compare against the comparand. The second value is the
                expression that follows the THEN.
            else_: An optional ELSE expression
        '''
        self.children = []  # type: List[EvaluatableNode]
        if not whens:
            raise ValueError("Must provide at least one WHEN for a CASE")
        for when, then in whens:
            if not isinstance(when, EvaluatableNode):
                raise ValueError("Invalid CASE expression; WHEN {} must be an expression."
                                 .format(comparand))
            if not isinstance(comparand, _EmptyNode):
                # If there's a comparand, generate conditions by comparing the
                # comparand with each WHEN
                # This allows us to consistently treat each when as a binary expression,
                # and not have to separately store the comparand.
                if not isinstance(comparand, EvaluatableNode):
                    raise ValueError("Invalid CASE expression; comparand {} must be an expression."
                                     .format(comparand))
                when = BinaryExpression(comparand, '=', when)
            self.children.append(when)
            self.children.append(then)
        # The fallback value, if no conditions are true, will be the ELSE if provided else NULL
        self.children.append(else_ if not isinstance(else_, _EmptyNode) else Value(None, None))

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        new_children = list(new_children)
        new_else_ = new_children.pop()
        return Case(
            comparand=EMPTY_NODE,
            whens=zip(new_children[::2], new_children[1::-2]),
            else_=new_else_)

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries

        # Evaluated_children has the following structure:
        # when0, then0, when1, then1, ... else
        # that is, a condition which, if true, is followed by what to return for that condition,
        # followed by another condition, etc.  The final value is what is returned when no
        # conditions are true.  We evaluate this as if it were a nested series of IF clauses,
        # i.e. IF(when0, then0, IF(when1, then1, ... IF(whenn, thenn, else_) ... ))
        #
        # First pop off the else column off the end.

        result = evaluated_children.pop()

        # Next, evaluate each nested IF/ELSE by walking backward through the whens
        for condition, then in zip(evaluated_children[-2::-2], evaluated_children[-1::-2]):
            if condition.type_ != BQScalarType.BOOLEAN:
                raise ValueError("CASE condition isn't boolean! Found: {!r}".format(condition))
            result = TypedSeries(then.series.where(condition.series, result.series),
                                 implicitly_coerce(then.type_, result.type_))
        return result


class Cast(MarkerSyntaxTreeNode, EvaluatableNodeWithChildren):
    '''An expression that converts an expression of one type to another type.  For example:
    CAST(1 AS STRING)
    '''
    def __init__(self, expression, type_):
        # type: (EvaluatableNode, str) -> None
        '''Set up Cast node

        Args:
            expression: The expression whose type to change
            type_: The type to cast to (string must resolve to BQScalarType)
        '''
        self.children = [expression]  # type: List[EvaluatableNode]
        self.type_ = BQScalarType.from_string(type_)

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return Cast(new_arguments[0], self.type_.value)

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        evaluated_expression = evaluated_children[0]
        return TypedSeries(
            evaluated_expression.series.astype(self.type_.to_dtype()), self.type_)


class Exists(MarkerSyntaxTreeNode, EvaluatableLeafNode):
    '''An expression that returns TRUE if the subquery produces one or more rows.  For example:
    EXISTS(SELECT a FROM table WHERE a=1)
    '''
    def __init__(self, subquery):
        # type: (DataframeNode) -> None
        '''An EXISTS node

        Args:
            subquery: A subquery
        '''
        self.subquery = subquery

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        return self

    def _evaluate_leaf_node(self, context):
        # type: (EvaluationContext) -> TypedSeries
        # We need to calculate the Exists for each row in the current
        # context's table.  We also need to provide the current context to the
        # Select query, because it can reference fields from the current
        # context.  For example:
        # SELECT a, EXISTS(SELECT * FROM table_b WHERE table_a.a = table_b.b)
        # FROM table_a
        # This returns a row with: `a` and whether there exists a `b` that
        # equals it.  The inner Select query needs to know about table_a in
        # order to compare `a` to `b`.
        results = []  # type: List[bool]
        for index, row in context.table.dataframe.iterrows():
            # Create a new context just for this one row
            single_row_df = TypedDataFrame(
                pd.DataFrame([row], index=context.table.dataframe.index), context.table.types)
            row_context = EvaluationContext.clone_context_new_table(single_row_df, context)
            typed_df, df_name = self.subquery.get_dataframe(context.datasets, row_context)
            results.append(len(typed_df.dataframe) > 0)
        # Construct a Series that contains each of the individual result rows
        return TypedSeries(pd.Series(results, index=_get_index(context.table.dataframe)),
                           BQScalarType.BOOLEAN)


class Extract(MarkerSyntaxTreeNode, EvaluatableNodeWithChildren):
    '''An expression that returns the value corresponding to the specified date part.
    For example:
    EXTRACT(DAY FROM timestamp)

    Reference:
    https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions#extract
    '''
    def __init__(self, part, date_expression):
        # type: (str, EvaluatableNode) -> None
        '''An EXTRACT node

        Args:
            part: The part of the date object to retrieve (day, month, year, etc)
            date_expression: The timestamp from which to extract the part
                (must evaluate to a pd.Timestamp)
        '''
        self.part = part.upper()
        self.children = [date_expression]

        weekdays = ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']
        valid_parts = (['DAYOFWEEK', 'DAY', 'DAYOFYEAR', 'WEEK', 'ISOWEEK',
                        'MONTH', 'QUARTER', 'YEAR', 'ISOYEAR'] +
                       ['WEEK({})'.format(day) for day in weekdays])
        if self.part not in valid_parts:
            raise ValueError('Not a valid date part to retrieve: {}'.format(self.part))

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return Extract(self.part, new_arguments[0])

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        date = evaluated_children[0].series
        # date_expression will evaluate to a pd.Timestamp, not datetime.datetime
        isoparts = ['ISOYEAR', 'ISOWEEK']
        if self.part in isoparts:
            # isocalendar() returns a tuple where first two elements are the
            # isoyear and the isoweek
            return TypedSeries(date.apply(lambda dt: dt.isocalendar()[isoparts.index(self.part)]),
                               BQScalarType.INTEGER)
        elif self.part.startswith('WEEK('):
            # WEEK(<weekday>) means the number of this week, assuming weeks
            # start with the given weekday.  There isn't an easy pandas analog
            # to this, and I don't think it's widely used
            raise NotImplementedError('{} not implemented'.format(self.part))
        else:
            return TypedSeries(date.apply(lambda dt: getattr(dt, self.part.lower())),
                               BQScalarType.INTEGER)


class If(MarkerSyntaxTreeNode, EvaluatableNodeWithChildren):
    '''A conditional expression, for example:

    IF(a > 0, a, 0)
        condition: a > 0
        then: a
        else_: 0
    Will return a if a is greater than zero, otherwise it will return zero.
    '''
    def __init__(self, condition, then, else_):
        # type: (EvaluatableNode, EvaluatableNode, EvaluatableNode) -> None
        '''Set up an If node

        Args:
            condition: A conditional expression, evaluates to a boolean
            then: Value to return if the condition is true
            else_: Value to return if the condition is false
        '''
        self.children = [condition, then, else_]

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return If(*new_arguments)

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        condition, then, else_ = evaluated_children
        if condition.type_ != BQScalarType.BOOLEAN:
            raise ValueError("IF condition isn't boolean! Found: {!r}".format(condition))
        result_type = implicitly_coerce(then.type_, else_.type_)
        return TypedSeries(then.series.where(condition.series, else_.series), result_type)


class InCheck(EvaluatableNodeWithChildren):
    '''Expression that checks whether element is in or not in a particular selection'''
    def __init__(self, expression, direction, elements):
        # type: (EvaluatableNode, str, Tuple[EvaluatableNode, ...]) -> None
        '''Set up InCheck node

        Args:
            expression: Expression to check
            direction: 'IN' or 'NOT_IN'
            elements: Set of values to be in (or not in)
        '''
        self.children = [expression] + list(elements)

        if direction == 'IN':
            self.direction = True
        elif direction == 'NOT_IN':
            self.direction = False
        else:
            raise ValueError("Invalid direction for InCheck, not IN or NOT_IN: {}"
                             .format(direction))

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return InCheck(new_children[0],
                       'IN' if self.direction else 'NOT_IN',
                       tuple(new_children[1:]))

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        expression_value = evaluated_children[0]
        element_values = evaluated_children[1:]
        contained = pd.Series([False] * len(expression_value.series))
        for element_value in element_values:
            contained = contained | (expression_value.series == element_value.series)
        if self.direction:
            return TypedSeries(contained, BQScalarType.BOOLEAN)
        else:
            # ~ is the boolean NOT operator
            return TypedSeries(~contained, BQScalarType.BOOLEAN)


class Not(MarkerSyntaxTreeNode, EvaluatableNodeWithChildren):
    '''Expression that negates the boolean series, such as turning [True] into [False]'''
    def __init__(self, expression):
        # type: (EvaluatableNode) -> None
        '''Set up NOT node

        Args:
            expression: The expression to negate (expression must evaluate to a series)
        '''
        self.children = [expression]

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return Not(new_arguments[0])

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        evaluated, = evaluated_children
        if not isinstance(evaluated, TypedSeries):
            raise ValueError("NOT expression must evaluate to a series")
        if evaluated.type_ != BQScalarType.BOOLEAN:
            raise ValueError("NOT accepts only booleans but was given type: {}".format(
                evaluated.type_))
        return TypedSeries(~evaluated.series, BQScalarType.BOOLEAN)


class NullCheck(EvaluatableNodeWithChildren):
    '''Expression that checks whether element is null or not'''

    def __init__(self, expression, direction):
        # type: (EvaluatableNode, str) -> None
        '''Set up NullCheck node

        Args:
            expression: Expression to check
            direction: 'IS_NULL' or 'IS_NOT_NULL'
        '''
        self.children = [expression]

        if direction == 'IS_NULL':
            self.direction = True
        elif direction == 'IS_NOT_NULL':
            self.direction = False
        else:
            raise ValueError("Invalid direction for NullCheck, not IS_NULL or IS_NOT_NULL: {}"
                             .format(direction))

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return NullCheck(new_arguments[0], 'IS_NULL' if self.direction else 'IS_NOT_NULL')

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        value = evaluated_children[0].series
        if self.direction:
            return TypedSeries(value.isnull(), BQScalarType.BOOLEAN)
        else:
            return TypedSeries(value.notnull(), BQScalarType.BOOLEAN)


class StarSelector(EvaluatableLeafNode):
    '''Corresponds a wildcard field of SELECT, like SELECT * or SELECT table.*'''

    StarSelectorType = Tuple[AbstractSyntaxTreeNode,  # expression
                             AbstractSyntaxTreeNode,  # exception
                             AbstractSyntaxTreeNode   # replacement
                             ]

    def __init__(self,
                 expression,  # type: AbstractSyntaxTreeNode
                 exception,  # type: AbstractSyntaxTreeNode
                 replacement,  # type: AbstractSyntaxTreeNode
                 ):
        # type: (...) -> None
        '''Sets up StarSelector node

        Args:
            selector: Columns to select from table.  Must be a tuple of (expression, exception,
            replacement).
        '''
        self.expression = expression
        self.exception = exception
        self.replacement = replacement

        # The position in the select list; 1-up (i.e. the first selector will have position=1).
        # This field will be populated by the Select object this Selector is passed to.
        self.position = None  # type: Optional[int]

    def name(self):
        # type: () -> str
        # It doesn't make sense to apply the .name() method to a select * or select table.*;
        # the point of .name is to return the name of a single column, but those constructions
        # return multiple columns.  Therefore it's an error to request the name of a select *.
        raise ValueError("select * doesn't have a name.")

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        raise ValueError("Can't evaluate select * in a group by context.")

    def _evaluate_leaf_node(self, context):
        # type: (EvaluationContext) -> TypedDataFrame
        '''See parent, EvaluatableNode'''

        if (self.expression is EMPTY_NODE and self.exception is EMPTY_NODE
                and self.replacement is EMPTY_NODE):
            # Bare SELECT *
            return context.table
        raise NotImplementedError("SELECT expression.* not implemented")


class Selector(EvaluatableNodeWithChildren):
    '''Corresponds to one of SELECT's fields, i.e. Selector is a column that
    should be returned from a SELECT query.
    Also includes the alias for this field, if specified.
    '''

    def __init__(self,
                 selector,  # type: EvaluatableNode
                 alias  # type: Union[_EmptyNode, str]
                 ):
        # type: (...) -> None
        '''Set up Selector node

        Args:
            selector: Column/expression to select from table.  Must be a tuple of
                an evaluatable expression.
            alias: Name to assign to this column/expression, if any.
        '''
        self.children = [selector]
        self.alias = alias

        # The position in the select list; 1-up (i.e. the first selector will have position=1).
        # This field will be populated by the Select object this Selector is passed to.
        self.position = None  # type: Optional[int]

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return Selector(new_children[0], self.name())

    def name(self):
        # type: () -> str
        if not isinstance(self.alias, _EmptyNode):
            return self.alias
        if self.position is None:
            raise RuntimeError("Accessing name of Selector {} before position is populated."
                               .format(self))
        return self.children[0].name() or '_f{}'.format(self.position)

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        if context.get_canonical_path((self.name(),)) in group_by_paths:
            return GroupedBy(self)
        return self.copy([self.children[0].mark_grouped_by(group_by_paths, context)])

    def _evaluate_node(self, evaluated_arguments):
        # type: (List[TypedSeries]) -> TypedSeries
        result, = evaluated_arguments
        result.series.name = self.name()
        return result


class UnaryNegation(EvaluatableNodeWithChildren):
    '''Expression that negates a series, such as turning [1, 2] into [-1, -2]'''
    def __init__(self, expression):
        # type: (EvaluatableNode) -> None
        ''' Set up unary negation

        Args:
            expression: The expression to negate; must evaluate to a series of
                either integers or floats.
        '''
        self.children = [expression]

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        typed_value, = evaluated_children
        if not isinstance(typed_value, TypedSeries):
            raise ValueError("UnaryNegation expression must evaluate to a series")
        if typed_value.type_ not in [BQScalarType.INTEGER, BQScalarType.FLOAT]:
            raise TypeError("UnaryNegation expression supports only integers and floats, got: {}"
                            .format(typed_value.type_))
        return TypedSeries(-typed_value.series, typed_value.type_)

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return UnaryNegation(new_arguments[0])


def _get_index(dataframe):
    # type: (Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]) -> pd.index
    '''Returns the index of a DataFrame.

    Args:
        dataframe: DataFrame or DataFrameGroupBy of data
    Returns:
        The index of the dataframe
    '''
    # Keep the same index as the rest of the expressions so that we can compare this constant
    # to other expressions.  If we're in a group by context, the index is the grouped by
    # column(s); to figure that out, apply some aggregating operation and get the index of the
    # aggregated result.  Different aggregating operations (.min, .max, .sum, etc.) would
    # return different results, obviously, but they'd all have the same index, which is what
    # we want here, so we just pick .max() arbitrarily.
    if isinstance(dataframe, pd.core.groupby.DataFrameGroupBy):
        return dataframe.max().index
    else:
        return dataframe.index


class Value(EvaluatableLeafNode):
    '''A node representing a literal value (number, string, boolean, null).'''

    def __init__(self, value, type_):
        # type: (Optional[LiteralType], Optional[BQScalarType]) -> None
        if (value is None) != (type_ is None):
            raise ValueError(
                "Value(None, None) means NULL; Value({}, {}) is not allowed"
                .format(value, type_))
        self.value = value
        self.type_ = type_

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        return self

    def strexpr(self):
        # type: () -> str
        return repr(self.value)

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, Value):
            return (self.type_ == other.type_) and (self.value == other.value)
        return False

    def _evaluate_leaf_node(self, context):
        # type: (EvaluationContext) -> TypedSeries
        '''See parent, EvaluatableNode'''
        return TypedSeries(pd.Series([self.value] * len(context.table.dataframe),
                                     index=_get_index(context.table.dataframe)),
                           self.type_)


_SeriesMaybeGrouped = Union[pd.Series, pd.core.groupby.SeriesGroupBy]
_FunctionType = Callable[[List[_SeriesMaybeGrouped]], _SeriesMaybeGrouped]
_OverClauseType = Tuple[Union[_EmptyNode, Sequence[EvaluatableNode]],
                        Union[_EmptyNode, List[Tuple[EvaluatableNode,
                                                     Union[_EmptyNode, str]]]]]


class _Function(object):
    '''Base class for functions.'''

    __metaclass__ = ABCMeta

    @classmethod
    def name(cls):
        # type: () -> str
        return cls.__name__.upper()

    # result_type=None means that the result type will be the same as the argument types.  e.g.
    # summing a column of floats gives a float, summing a column of ints gives an int.
    result_type = None  # type: Optional[BQType]


class _NonAggregatingFunction(_Function):
    '''Base class for regular functions (not aggregating).'''

    @abstractmethod
    def function(self, values):
        # type: (List[pd.Series]) -> pd.Series
        '''Computes a column from a list of argument columns.

        Args:
            values: A list of Pandas Serieses, i.e. columns of values to operate on.
        Returns:
            A single column of values as a Pandas Series.
        '''


class _AggregatingFunction(_Function):
    '''Base class for aggregating functions.'''

    @abstractmethod
    def aggregating_function(self, values):
        # type: (List[pd.Series]) -> LiteralType
        '''Collapses columns of values into a single number.

        Args:
            values: A list of Pandas Serieses, i.e. columns of values to operate on.

        Returns:
            A single Python value computed from the input arguments.
        '''


_CounteeType = Union[str,                           # a literal * i.e. COUNT(*)
                     Tuple[Union[str, _EmptyNode],  # optional modifier, e.g. DISTINCT
                           EvaluatableNode]]        # the thing counted.


class Count(_AggregatingFunction):
    '''A COUNT function, for example:
    SELECT COUNT(*) FROM Table
    '''

    result_type = BQScalarType.INTEGER

    @classmethod
    def create_count_function_call(cls,
                                   countee,  # type: _CounteeType
                                   over_clause  # type: _OverClauseType
                                   ):
        # type: (...) -> EvaluatableNode
        '''Creates a Count expression.

        COUNT has a factory creation method, unlike other _Function subtypes, because it has
        unique syntax (COUNT(*), COUNT(DISTINCT expr)), and because it has state in addition
        to its child nodes (whether or not to count only distinct rows).  This method is
        called by the grammar to create a FunctionCall syntax tree node of the appropriate type
        (aggregating or analytic) with a Count expression having the appropriate state.

        Args:
            countee: Either a single string '*' or a tuple: an optional
                string 'DISTINCT' and a required expression to count
            over_clause: An optional OVER clause
        Returns:
            An AggregatingFunctionCall expression, if the over clause isn't present,
            or an _AnalyticFunctionCall expression, if it is, either way using this function.
        '''
        # Treat count(*) as if it were count(1), which is equivalent.
        if isinstance(countee, str):
            if countee != '*':
                raise ValueError("Invalid expression COUNT({})".format(countee))
            countee = (EMPTY_NODE, Value(1, BQScalarType.INTEGER))
        maybe_distinct, argument = countee
        if maybe_distinct == 'DISTINCT':
            distinct = True
        elif maybe_distinct == EMPTY_NODE:
            distinct = False
        else:
            raise NotImplementedError("Non-DISTINCT modifiers for COUNT are not implemented:"
                                      " {}".format(maybe_distinct))

        if isinstance(over_clause, _EmptyNode):
            return _AggregatingFunctionCall(Count(distinct), [argument])
        else:
            return _AnalyticFunctionCall(Count(distinct), [argument], over_clause)

    def __init__(self, distinct):
        # type: (bool) -> None
        self.distinct = distinct

    def aggregating_function(self, values):
        # type: (List[pd.Series]) -> int
        value, = values
        value = value.dropna()  # COUNT counts non-NULL rows
        if self.distinct:
            value = set(value)
        return len(value)


class Mod(_NonAggregatingFunction):
    '''The modulus of two columns of numbers, i.e. remainder after a is divided by b.'''

    def function(self, values):
        a, b = values
        return a.mod(b)


class Sum(_AggregatingFunction):
    '''The sum of a column.'''

    def aggregating_function(self, values):
        # type: (List[pd.Series]) -> LiteralType
        value, = values
        return value.sum()


class Max(_AggregatingFunction):
    '''The maximum value of a column.'''

    def aggregating_function(self, values):
        # type: (List[pd.Series]) -> LiteralType
        value, = values
        return value.max()


class Min(_AggregatingFunction):
    '''The minimum value of a column.'''

    def aggregating_function(self, values):
        # type: (List[pd.Series]) -> LiteralType
        value, = values
        return value.min()


class Concat(_NonAggregatingFunction):
    '''The concatenation of a series of strings.'''

    result_type = BQScalarType.STRING

    def function(self, values):
        # type: (List[pd.Series]) -> pd.Series
        return reduce(operator.add, values)


class Timestamp(_NonAggregatingFunction):
    '''The conversion of a column of values to timestamps.'''

    result_type = BQScalarType.TIMESTAMP

    def function(self, values):
        # type: (List[pd.Series]) -> pd.Series
        return values[0].apply(pd.Timestamp)


class Row_Number(_NonAggregatingFunction):
    '''A numbering of the rows in a column.'''

    result_type = BQScalarType.INTEGER

    def function(self, values):
        # type: (List[pd.Series]) -> pd.Series
        value, = values
        return pd.Series(range(1, len(value) + 1), index=_get_index(value))


class FunctionCall(object):
    '''Abstract base class for function call expressions. Subclasses are aggregating or not.'''

    _FUNCTION_MAP = {function_info.name(): function_info()
                     for function_info in (Min, Max, Sum, Mod, Concat, Timestamp, Row_Number)}

    @classmethod
    def create(cls,
               function_name,  # type: str
               expression,  # type: Union[_EmptyNode, List[EvaluatableNode]]
               over_clause  # type: Union[_EmptyNode, _OverClauseType]
               ):
        # type: (...) -> EvaluatableNode
        function_name = function_name.upper()
        if function_name not in cls._FUNCTION_MAP:
            raise NotImplementedError('Function {} not implemented'.format(function_name))
        function_info = cls._FUNCTION_MAP[function_name]

        # The implementation of aggregating functions currently requires exactly one argument.
        # This works for sum, min, max; it excludes a few functions like corr.  The reason
        # for the restriction is that in a group by context, the aggregating function is called
        # by applying it to a single column.  Lifting the restriction will require combining
        # evaluated SeriesGroupBys into a single DataFrameGroupBy.
        if (isinstance(function_info, _AggregatingFunction) and
                (isinstance(expression, _EmptyNode) or len(expression) != 1)):
            raise NotImplementedError("Aggregating functions are only supported with 1 argument")

        if not isinstance(over_clause, _EmptyNode):
            return _AnalyticFunctionCall(function_info, expression, over_clause)
        elif isinstance(function_info, _AggregatingFunction):
            return _AggregatingFunctionCall(function_info, expression)
        elif isinstance(function_info, _NonAggregatingFunction):
            return _NonAggregatingFunctionCall(function_info, expression)
        else:
            raise ValueError("Invalid function info {}".format(function_info))

    def _evaluate(self, arguments, function, result_type):
        # type: (List[TypedSeries], _FunctionType, BQType) -> TypedSeries
        '''Evaluates arguments using function.

        Args:
            arguments: A list of TypedSeries (columns of data)
            function: A function to apply to the arguments.
            result_type: Type of the result of this function call.  None means depends on the
                arguments.
        Returns:
            The result of applying the function to the arguments.
        '''
        argument_values = [argument.series for argument in arguments]
        argument_types = [argument.type_ for argument in arguments]
        if result_type is None:
            result_type = implicitly_coerce(*argument_types)
        return TypedSeries(function(argument_values), result_type)


class _NonAggregatingFunctionCall(FunctionCall, EvaluatableNodeWithChildren):
    '''A function call that does not aggregate rows, e.g. concat.'''

    def __init__(self, function_info, expression):
        # type: (_NonAggregatingFunction, Union[_EmptyNode, Sequence[EvaluatableNode]]) -> None
        self.function_info = function_info
        if not isinstance(expression, _EmptyNode):
            self.children = expression
        else:
            self.children = []

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return _NonAggregatingFunctionCall(self.function_info, new_arguments)

    def _evaluate_node(self, arguments):
        # type: (List[TypedSeries]) -> TypedSeries
        return self._evaluate(
                arguments, self.function_info.function, self.function_info.result_type)


class _AggregatingFunctionCall(FunctionCall, EvaluatableNodeThatAggregatesOrGroups):
    '''A function call that aggregates rows, e.g. sum.'''

    def __init__(self, function_info, expression):
        # type: (_AggregatingFunction, Union[_EmptyNode, Sequence[EvaluatableNode]]) -> None
        self.function_info = function_info
        if not isinstance(expression, _EmptyNode):
            self.children = expression
        else:
            self.children = []

    def copy(self, new_arguments):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return _AggregatingFunctionCall(self.function_info, new_arguments)

    def _evaluate_node(self, arguments):
        # type: (List[TypedSeries]) -> TypedSeries
        return self._evaluate(
            arguments,
            lambda values: pd.Series(self.function_info.aggregating_function(values)),
            self.function_info.result_type)

    def _evaluate_node_in_group_by(self, arguments):
        # type: (List[TypedSeries]) -> TypedSeries
        return self._evaluate(
            arguments,
            lambda values: values[0].apply(lambda x: self.function_info.aggregating_function([x])),
            self.function_info.result_type)


class _AnalyticFunctionCall(FunctionCall, EvaluatableNodeWithChildren):
    '''A function call that is evaluated over windows of the data but with results for each row.'''

    def __init__(self, function_info, maybe_arguments, over_clause):
        # type: (_Function, Union[_EmptyNode, Sequence[EvaluatableNode]], _OverClauseType) -> None
        '''Creates a function call expression for an invocation of an analytic function

        Args:
            function_info: A subtype of _Function defining the specific function to evaluate
                over this window.
            maybe_arguments: Either EMPTY_NODE (if the function is called with no arguments,
                i.e. function_name()) or a list of syntax tree nodes giving the arguments.
            over_clause: The clause defining the window over which the function is evaluated.
                This is a tuple of two parts, both optional (i.e. EMPTY_NODE if not provided):
                  - a list of expressions by which to partition the set of rows
                  - a list of expressions by which to sort the set of rows.
        '''
        self.function_info = function_info
        maybe_partition_by, maybe_order_by = over_clause
        partition_by = [] if isinstance(maybe_partition_by, _EmptyNode)else list(maybe_partition_by)
        order_by = [] if isinstance(maybe_order_by, _EmptyNode) else list(maybe_order_by)
        self.order_by_ascending = [
            isinstance(direction, _EmptyNode) or direction.upper() != 'DESC'
            for _, direction in order_by]
        arguments = [] if isinstance(maybe_arguments, _EmptyNode) else list(maybe_arguments)

        # Row_number() doesn't take any arguments, but it does need to know how many rows there are.
        # An easy way to make that work is to pass a constant 1 argument, which will be expanded
        # to the number of rows.
        #
        # When more analytic functions are implemented, find a general way to specify
        # pseudo-arguments; e.g. rank() and dense_rank() will require passing something like
        # pd.factorize(zip(order_by)), in order to know which rows compare equal.
        if function_info.name() == 'ROW_NUMBER':
            arguments = [Value(1, BQScalarType.INTEGER)]

        self.children = arguments + partition_by + [expr for expr, _ in order_by]
        self.num_arguments = len(arguments)
        self.num_partition_by = len(partition_by)

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        return _AnalyticFunctionCall(
                self.function_info,
                new_children[:self.num_arguments],
                (new_children[self.num_arguments:self.num_arguments + self.num_partition_by],
                 [(argument, 'ASC' if ascending else 'DESC')
                  for argument, ascending in zip(
                          new_children[self.num_arguments + self.num_partition_by:],
                          self.order_by_ascending)]))

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries

        # To evaluate an analytic expression, we perform something like an entire
        # select/order by/group by, except it results in a single column.
        #
        # Start with a completely empty evaluation context.
        context = EvaluationContext({})
        context.table = TypedDataFrame(pd.DataFrame(), [])

        # Add into the context the expressions referenced as function arguments, ORDER BY and
        # PARTITION BY parameters.  We store the canonical identifier path to each one.
        paths = [context.maybe_add_column(child) for child in evaluated_children]

        # Split up the paths into those three segments.
        argument_paths = paths[:self.num_arguments]
        partition_by_paths = paths[self.num_arguments:self.num_arguments+self.num_partition_by]
        order_by_paths = paths[self.num_arguments + self.num_partition_by:]

        # Execute the order by, if present: sort the context
        if order_by_paths:
            context.table = TypedDataFrame(
                    context.table.dataframe.sort_values(by=['.'.join(path)
                                                            for path in order_by_paths],
                                                        ascending=self.order_by_ascending),
                    context.table.types)

        # Execute the partition by.  If it's present, we will evaluate over a partition
        # (grouping) of the rows.  Otherwise, we group all the rows together into a single group.
        if partition_by_paths:
            grouped_table = context.table.dataframe.groupby(
                ['.'.join(path) for path in partition_by_paths])
        else:
            # If no PARTITION BY is specified, all rows are put in a single group.
            context.table.dataframe['__constant__'] = 100
            grouped_table = context.table.dataframe.groupby('__constant__')
        context.table = TypedDataFrame(grouped_table, context.table.types)

        # Now extract just the function arguments from the (sorted, grouped) context
        evaluated_arguments = [context.lookup(path) for path in argument_paths]

        # Calculate the result type (just as is done for the other FunctionCall types)
        result_type = self.function_info.result_type
        if result_type is None:
            result_type = implicitly_coerce(*[argument.type_ for argument in evaluated_arguments])

        # Determine which function to call
        if isinstance(self.function_info, _AggregatingFunction):
            function = self.function_info.aggregating_function
        elif isinstance(self.function_info, _NonAggregatingFunction):
            function = self.function_info.function
        else:
            raise RuntimeError("Invalid function info {}".format(self.function_info))

        # Call the function on each group, calculating a value for each row.  Unlike an aggregation,
        # the output size is the same as the input size.
        #
        # We extract just the first argument because .transform works on individual columns.
        # TODO: figure out how to apply it to multiple columns simultaneously, which would enable
        # multi-column aggregators like CORR.
        evaluated_argument, = evaluated_arguments
        result = evaluated_argument.series.transform(lambda x: function([x]))
        return TypedSeries(result, result_type)
