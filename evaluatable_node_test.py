import datetime
import unittest
from re import escape
from typing import Any, List, Optional, Tuple, Union  # noqa: F401

import numpy as np
import pandas as pd
import six
from ddt import data, ddt, unpack

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import (EMPTY_CONTEXT, EMPTY_NODE, EvaluatableNode,  # noqa: F401
                                     EvaluationContext, Field, GroupedBy, _EmptyNode)
from bq_types import BQScalarType, BQType, PythonType, TypedDataFrame, TypedSeries  # noqa: F401
from dataframe_node import TableReference
from evaluatable_node import (Case, Cast, Exists, Extract, FunctionCall, If, InCheck, Not,
                              NullCheck, Selector, UnaryNegation, Value)
from grammar import select as select_rule
from query_helper import apply_rule
from tokenizer import tokenize


@ddt
class EvaluatableNodeTest(unittest.TestCase):

    def setUp(self):
        # type: () -> None
        self.small_datasets = {
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1], [2]], columns=['a']),
                        types=[BQScalarType.INTEGER]
                    )
                }
            }
        }

        self.large_datasets = {
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 2, 3], [1, 4, 3]], columns=['a', 'b', 'c']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        }

    def test_selector(self):
        # type: () -> None
        selector = Selector(Field(('a',)), 'field_alias')
        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = selector.evaluate(context)

        self.assertEqual(list(dataframe.series), [[1], [2]])
        self.assertEqual(list(dataframe.dataframe), ['field_alias'])
        self.assertEqual(dataframe.types, [BQScalarType.INTEGER])

    def test_selector_group_by_success(self):
        # type: () -> None
        selector = Selector(Field(('c',)), EMPTY_NODE)
        selector.position = 1
        context = EvaluationContext(self.large_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)

        context.exclude_aggregation = True
        updated_selector, = context.do_group_by([selector], [Field(('my_table', 'c'))])

        dataframe = updated_selector.evaluate(context)

        self.assertEqual(list(dataframe.series), [3])

    @data((5, BQScalarType.INTEGER),
          (1.23, BQScalarType.FLOAT),
          ("something", BQScalarType.STRING),
          (True, BQScalarType.BOOLEAN),
          (None, None))
    @unpack
    def test_value_repr(self, value, type_):
        # type: (PythonType, Optional[BQType]) -> None
        '''Check Value's string representation'''
        node = Value(value, type_)
        representation = 'Value(type_={}, value={})'.format(type_.__repr__(), value.__repr__())
        self.assertEqual(node.__repr__(), representation)

    def test_value_eval(self):
        # type: () -> None
        # A constant is repeated for each row in the context table.
        value = Value(12345, BQScalarType.INTEGER)
        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')), 'foo')
        dataframe = value.evaluate(context)
        self.assertEqual(list(dataframe.series), [12345, 12345])

    def test_field(self):
        # type: () -> None
        field = Field(('a',))
        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = field.evaluate(context)
        self.assertEqual(list(dataframe.series), [[1], [2]])
        self.assertEqual(dataframe.series.name, 'a')

    @data(
        dict(function_name='sum', args=[Field(('a',))], expected_result=[3]),
        dict(function_name='max', args=[Field(('a',))], expected_result=[2]),
        dict(function_name='min', args=[Field(('a',))], expected_result=[1]),
        dict(function_name='concat',
             args=[Value('foo', BQScalarType.STRING), Value('bar', BQScalarType.STRING)],
             expected_result=['foobar'] * 2),  # two copies to match length of context table.
        dict(function_name='mod',
             args=[Field(('a',)), Value(2, BQScalarType.INTEGER)],
             expected_result=[1, 0]),
        dict(function_name='mod',
             args=[Value(1.0, BQScalarType.FLOAT), Value(2, BQScalarType.INTEGER)],
             expected_result=[1.0, 1.0]),
        dict(function_name='timestamp',
             args=[Value("2019-04-22", BQScalarType.STRING)],
             expected_result=[datetime.datetime(2019, 4, 22)] * 2),  # two copies to match table len
    )
    @unpack
    def test_functions(self, function_name, args, expected_result):
        # type: (str, List[EvaluatableNode], List[PythonType]) -> None
        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        result = FunctionCall.create(function_name, args, EMPTY_NODE).evaluate(context)
        self.assertEqual(
                [result.type_.convert(elt) for elt in result.series],
                expected_result)

    def test_bad_function(self):
        # type: () -> None
        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        with self.assertRaisesRegexp(NotImplementedError, 'NOT_A_FUNCTION not implemented'):
            FunctionCall.create('not_a_function', [], EMPTY_NODE).evaluate(context)

    @data(
        # Explore each aggregate function, along with a non-aggregate function to make sure we
        # can compute both at once.
        dict(selectors='sum(a), b+10', expected_result=[[6, 11], [5, 12]]),
        dict(selectors='sum(a+1), b+10', expected_result=[[8, 11], [6, 12]]),
        dict(selectors='max(a), b+10', expected_result=[[4, 11], [5, 12]]),
        dict(selectors='min(a), b+10', expected_result=[[2, 11], [5, 12]]),
        dict(selectors='count(a), b+10', expected_result=[[2, 11], [1, 12]]),
        dict(selectors='count(*), b+10', expected_result=[[2, 11], [2, 12]]),
    )
    @unpack
    def test_aggregate_functions_in_group_by(self, selectors, expected_result):
        # type: (str, List[List[int]]) -> None
        datasets = {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame([[2, 1], [4, 1], [5, 2], [np.nan, 2]], columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER])}}}

        tokens = tokenize('select {} from my_table group by b'.format(selectors))
        node, leftover = select_rule(tokens)
        result, unused_table_name = node.get_dataframe(datasets)
        self.assertFalse(leftover)
        self.assertEqual(result.to_list_of_lists(), expected_result)

    @data(
        # Row number over whole dataset; order is not guaranteed
        dict(selectors='row_number() over ()', expected_result=[[1], [2], [3], [4]]),
        dict(selectors='row_number() over (order by a), a',
             expected_result=[[1, 10], [2, 20], [3, 30], [4, 30]]),
        dict(selectors='row_number() over (order by a asc), a',
             expected_result=[[1, 10], [2, 20], [3, 30], [4, 30]]),
        dict(selectors='row_number() over (order by a desc), a',
             expected_result=[[4, 10], [3, 20], [2, 30], [1, 30]]),
        dict(selectors='row_number() over (partition by b order by a), a',
             expected_result=[[1, 10], [2, 20], [1, 30], [2, 30]]),
        dict(selectors='sum(a) over (), a',
             expected_result=[[90, 10], [90, 20], [90, 30], [90, 30]]),
        dict(selectors='sum(a) over (partition by b), a',
             expected_result=[[30, 10], [30, 20], [60, 30], [60, 30]]),
        dict(selectors='count(*) over (), a',
             expected_result=[[4, 10], [4, 20], [4, 30], [4, 30]]),
        dict(selectors='count(a) over (), a',
             expected_result=[[4, 10], [4, 20], [4, 30], [4, 30]]),
        dict(selectors='count(*) over (partition by b), a',
             expected_result=[[2, 10], [2, 20], [2, 30], [2, 30]]),
        dict(selectors='count(a) over (partition by b), a',
             expected_result=[[2, 10], [2, 20], [2, 30], [2, 30]]),
        dict(selectors='sum(count(*)) over ()',
             expected_result=[[4]]),
    )
    @unpack
    def test_analytic_function(self, selectors, expected_result):
        datasets = {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame([[20, 200], [10, 200], [30, 300], [30, 300]], columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER])}}}
        tokens = tokenize('select {} from my_table'.format(selectors))
        node, leftover = select_rule(tokens)
        result, unused_table_name = node.get_dataframe(datasets)
        self.assertFalse(leftover)
        # Note: BQ docs say if ORDER BY clause (for the select as a whole) is not present, order of
        # results is undefined, so we do not assert on the order.
        six.assertCountEqual(self, result.to_list_of_lists(), expected_result)

    @data(
        dict(selectors='sum(count(*)) over (), count(*)',
             expected_result=[[5, 2], [5, 3]]),
    )
    @unpack
    def test_analytic_function_with_group_by(self, selectors, expected_result):
        datasets = {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame([[20, 2], [10, 2], [30, 3], [31, 3], [32, 3]], columns=['a', 'b']),
                types=[BQScalarType.INTEGER, BQScalarType.INTEGER])}}}
        tokens = tokenize('select {} from my_table group by b'.format(selectors))
        node, leftover = select_rule(tokens)
        result, unused_table_name = node.get_dataframe(datasets)
        self.assertFalse(leftover)
        # Note: BQ docs say if ORDER BY clause (for the select as a whole) is not present, order of
        # results is undefined, so we do not assert on the order.
        six.assertCountEqual(self, result.to_list_of_lists(), expected_result)

    def test_non_aggregate_function_in_group_by(self):
        datasets = {'my_project': {'my_dataset': {'my_table': TypedDataFrame(
                pd.DataFrame([['one', '1'], ['two', '1'], ['three', '2'], ['four', '2']],
                             columns=['a', 'b']),
                types=[BQScalarType.STRING, BQScalarType.INTEGER])}}}

        tokens = tokenize('select max(concat(b, "hi")) from my_table group by b')
        node, leftover = select_rule(tokens)
        self.assertFalse(leftover)
        result, unused_table_name = node.get_dataframe(datasets)
        self.assertEqual(result.to_list_of_lists(), [['1hi'], ['2hi']])

    @data(
        dict(count='COUNT(*)', expected_result=[[2]]),
        dict(count='COUNT(c)', expected_result=[[2]]),
        dict(count='COUNT(DISTINCT c)', expected_result=[[1]]),
        dict(count='COUNT(b)', expected_result=[[2]]),
        dict(count='COUNT(DISTINCT b)', expected_result=[[2]]),
        dict(count='COUNT(a)', expected_result=[[1]]),
    )
    @unpack
    def test_count(self, count, expected_result):
        # type: (str, List[List[int]]) -> None
        count_datasets = {
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, 2, 3], [None, 4, 3]], columns=['a', 'b', 'c']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        }
        select, leftover = select_rule(tokenize('SELECT {} FROM my_table'.format(count)))
        self.assertFalse(leftover)
        dataframe, unused_table_name = select.get_dataframe(count_datasets)
        self.assertEqual(dataframe.to_list_of_lists(), expected_result)

    @data(('IS_NULL', [True, False]), ('IS_NOT_NULL', [False, True]))
    @unpack
    def test_null_check(self, direction, result):
        # type: (str, List[bool]) -> None
        datasets = {
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1, None], [2, 3]], columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    )
                }
            }
        }

        context = EvaluationContext(datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        expression = Field(('b',))
        null_check = NullCheck(expression, direction)

        dataframe = null_check.evaluate(context)
        self.assertEqual(list(dataframe.series), result)

    @data(('IN', [True, False]), ('NOT_IN', [False, True]))
    @unpack
    def test_in_check(self, direction, result):
        # type: (str, List[bool]) -> None
        expression = Field(('a',))
        elements = (Value(1, type_=BQScalarType.INTEGER), Value(3, type_=BQScalarType.INTEGER))
        in_check = InCheck(expression, direction, elements)

        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = in_check.evaluate(context)
        self.assertEqual(list(dataframe.series), result)

    @data(
        (True, 0),
        (False, 1)
    )
    @unpack
    def test_if_empty_context(self, condition_bool, result):
        # type: (bool, int) -> None
        condition = Value(condition_bool, BQScalarType.BOOLEAN)
        then = Value(0, BQScalarType.INTEGER)
        else_ = Value(1, BQScalarType.INTEGER)
        # IF [condition] THEN 0 ELSE 1
        if_expression = If(condition, then, else_)

        dataframe = if_expression.evaluate(EMPTY_CONTEXT)
        self.assertEqual(list(dataframe.series), [result])

    def test_if(self):
        condition = BinaryExpression(Field(('a',)), '>', Value(1, BQScalarType.INTEGER))
        then = Value('yes', BQScalarType.STRING)
        else_ = Value('no', BQScalarType.STRING)
        # IF a > 1 THEN "yes" ELSE "no"
        if_expression = If(condition, then, else_)

        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = if_expression.evaluate(context)
        self.assertEqual(list(dataframe.series), ['no', 'yes'])

    def test_if_different_types(self):
        condition = Value(True, BQScalarType.BOOLEAN)
        then = Value('yes', BQScalarType.STRING)
        else_ = Value(1, BQScalarType.INTEGER)
        if_expression = If(condition, then, else_)

        error = "Cannot implicitly coerce the given types: " \
                "\(BQScalarType.STRING, BQScalarType.INTEGER\)"
        with self.assertRaisesRegexp(ValueError, error):
            if_expression.evaluate(EMPTY_CONTEXT)

    def test_if_error(self):
        condition = Value(5, BQScalarType.INTEGER)
        then = Value(0, BQScalarType.INTEGER)
        else_ = Value(1, BQScalarType.INTEGER)
        if_expression = If(condition, then, else_)

        error = escape("IF condition isn't boolean! Found: {}".format(
            str(condition.evaluate(EMPTY_CONTEXT))))
        with self.assertRaisesRegexp(ValueError, error):
            if_expression.evaluate(EMPTY_CONTEXT)

    def test_not(self):
        expression = Value(True, BQScalarType.BOOLEAN)
        not_expression = Not(expression)

        dataframe = not_expression.evaluate(EMPTY_CONTEXT)
        self.assertEqual(list(dataframe.series), [False])

    def test_not_type_error(self):
        expression = Value(5, BQScalarType.INTEGER)
        not_expression = Not(expression)

        with self.assertRaisesRegexp(ValueError, ""):
            not_expression.evaluate(EMPTY_CONTEXT)

    @data(
        (1, BQScalarType.INTEGER, -1),
        (1.0, BQScalarType.FLOAT, -1.0),
    )
    @unpack
    def test_unary_negation(self, initial_value, value_type, result_value):
        # type: (Any, BQScalarType, Any) -> None
        expression = Value(initial_value, value_type)
        negation = UnaryNegation(expression)

        dataframe = negation.evaluate(EMPTY_CONTEXT)
        self.assertEqual(list(dataframe.series), [result_value])

    @data(
        ("abc", BQScalarType.STRING),
        (True, BQScalarType.BOOLEAN),
    )
    @unpack
    def test_unary_negation_error(self, value, value_type):
        # type: (Any, BQScalarType) -> None
        expression = Value(value, value_type)
        negation = UnaryNegation(expression)

        error = ("UnaryNegation expression supports only integers and floats, got: {}"
                 .format(value_type))
        with self.assertRaisesRegexp(TypeError, error):
            negation.evaluate(EMPTY_CONTEXT)

    @data(
        dict(
            comparand=Field(('a',)),
            whens=[(Value(1, BQScalarType.INTEGER), Value("one", BQScalarType.STRING)),
                   (Value(2, BQScalarType.INTEGER), Value("two", BQScalarType.STRING))],
            else_=Value("other", BQScalarType.STRING),
            result=["one", "two"]
        ),
        dict(
            comparand=Field(('a',)),
            whens=[(Value(1, BQScalarType.INTEGER), Value("one", BQScalarType.STRING))],
            else_=Value("other", BQScalarType.STRING),
            result=["one", "other"]
        ),
        dict(
            comparand=EMPTY_NODE,
            whens=[(Value(True, BQScalarType.BOOLEAN), Value("yes", BQScalarType.STRING)),
                   (Value(False, BQScalarType.BOOLEAN), Value("no", BQScalarType.STRING))],
            else_=EMPTY_NODE,
            result=["yes", "yes"]
        ),
        dict(
            comparand=Field(('a',)),
            whens=[(Value(1, BQScalarType.INTEGER), Value("one", BQScalarType.STRING))],
            else_=EMPTY_NODE,
            result=["one", None]
        ),
    )
    @unpack
    def test_case_with_comparand(self, comparand,  # type: Union[_EmptyNode, EvaluatableNode]
                                 whens,  # type: List[Tuple[EvaluatableNode, EvaluatableNode]]
                                 else_,  # type: EvaluatableNode
                                 result  # type: List[str]
                                 ):
        # type: (...) -> None
        case = Case(comparand, whens, else_)

        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = case.evaluate(context)
        self.assertEqual(list(dataframe.series), result)

    def test_case_no_whens(self):
        comparand = EMPTY_NODE
        whens = []
        else_ = EMPTY_NODE

        error = "Must provide at least one WHEN for a CASE"
        with self.assertRaisesRegexp(ValueError, error):
            Case(comparand, whens, else_)

    @data(
        dict(
            comparand=EMPTY_NODE,
            whens=[(Value(1, BQScalarType.INTEGER), Value("one", BQScalarType.STRING))],
            else_=EMPTY_NODE,
            error="CASE condition isn't boolean! Found: {!r}".format(
                TypedSeries(pd.Series([1, 1]), BQScalarType.INTEGER))
        ),
        dict(
            comparand=Field(('a',)),
            whens=[(Value(1, BQScalarType.INTEGER), Value("one", BQScalarType.STRING))],
            else_=Value(100, BQScalarType.INTEGER),
            error="Cannot implicitly coerce the given types: "
                  "(BQScalarType.STRING, BQScalarType.INTEGER)"
        ),
    )
    @unpack
    def test_case_error(self, comparand,  # type: Union[_EmptyNode, EvaluatableNode]
                        whens,  # type: List[Tuple[EvaluatableNode, EvaluatableNode]]
                        else_,  # type: EvaluatableNode
                        error  # type: str
                        ):
        # type: (...) -> None
        case = Case(comparand, whens, else_)

        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        with self.assertRaisesRegexp(ValueError, escape(error)):
            case.evaluate(context)

    @data(
        dict(
            value=Value(1, BQScalarType.INTEGER),
            cast_type='STRING',
            result='1'
        ),
        dict(
            value=Value(1, BQScalarType.INTEGER),
            cast_type='FLOAT',
            result=1.0
        ),
        dict(
            value=Value(1, BQScalarType.INTEGER),
            cast_type='BOOLEAN',
            result=True
        ),
        dict(
            value=Value(1.0, BQScalarType.FLOAT),
            cast_type='STRING',
            result='1.0'
        ),
        dict(
            value=Value(1.0, BQScalarType.FLOAT),
            cast_type='INTEGER',
            result=1
        ),
        dict(
            value=Value(True, BQScalarType.BOOLEAN),
            cast_type='STRING',
            result='True'
        ),
        dict(
            value=Value(True, BQScalarType.BOOLEAN),
            cast_type='INTEGER',
            result=1
        ),
        dict(
            value=Value('1', BQScalarType.STRING),
            cast_type='INTEGER',
            result=1
        ),
        dict(
            value=Value('1.0', BQScalarType.STRING),
            cast_type='FLOAT',
            result=1.0
        ),
        dict(
            value=Value('TRUE', BQScalarType.STRING),
            cast_type='BOOLEAN',
            result=True
        ),
        dict(
            value=Value('2019-12-01', BQScalarType.STRING),
            cast_type='DATETIME',
            result=datetime.datetime(2019, 12, 1),
        ),
        dict(
            value=Value('2019-12-01', BQScalarType.STRING),
            cast_type='DATE',
            result=datetime.date(2019, 12, 1),
        ),
        dict(
            value=Value('2019-12-01 01:02:03', BQScalarType.STRING),
            cast_type='TIMESTAMP',
            result=datetime.datetime(2019, 12, 1, 1, 2, 3),
        ),
        dict(
            value=Value(pd.Timestamp('2019-12-01'), BQScalarType.DATE),
            cast_type='DATETIME',
            result=datetime.datetime(2019, 12, 1),
        ),
        dict(
            value=Value(pd.Timestamp('2019-12-01'), BQScalarType.DATE),
            cast_type='TIMESTAMP',
            result=datetime.datetime(2019, 12, 1),
        ),
        dict(
            value=Value(pd.Timestamp('2019-12-01 00:01:02'), BQScalarType.DATETIME),
            cast_type='DATE',
            result=datetime.date(2019, 12, 1),
        ),
        dict(
            value=Value(pd.Timestamp('2019-12-01 00:01:02'), BQScalarType.DATETIME),
            cast_type='TIMESTAMP',
            result=datetime.datetime(2019, 12, 1, 0, 1, 2),
        ),
    )
    @unpack
    def test_cast(self, value, cast_type, result):
        # type: (Value, str, pd.Timestamp) -> None
        cast = Cast(value, cast_type)

        series = cast.evaluate(EMPTY_CONTEXT)
        self.assertEqual(series.to_list(), [result])

    @data(
        dict(
            value=Value("abc", BQScalarType.STRING),
            cast_type=BQScalarType.INTEGER,
            # TODO: This error message should be about converting to
            # int, not float.  But bq_types currently defines
            # BQScalarType.INTEGER converting to float64.
            #
            # Python 3 surrounds the expression with quotes, Python 2 doesn't, so the
            # regex .? matches the Py3-only quote.
            error="could not convert string to float: .?abc"
        ),
        dict(
            value=Value("abc", BQScalarType.STRING),
            cast_type=BQScalarType.FLOAT,
            # Python 3 surrounds the expression with quotes, Python 2 doesn't, so the
            # regex .? matches the Py3-only quote.
            error="could not convert string to float: .?abc"
        ),
        dict(
            value=Value("abc", BQScalarType.STRING),
            cast_type=BQScalarType.TIMESTAMP,
            error="Error parsing datetime string \"abc\" at position 0"
        ),
    )
    @unpack
    def test_cast_error(self, value, cast_type, error):
        # type: (Value, BQScalarType, str) -> None
        cast = Cast(value, cast_type)

        with self.assertRaisesRegexp(ValueError, error):
            cast.evaluate(EMPTY_CONTEXT)

    @data(
        ("select a from `my_project.my_dataset.my_table` where a=1", [True, True]),
        ("select a from `my_project.my_dataset.my_table` where a=10", [False, False]),
    )
    @unpack
    def test_exists(self, select_query, result):
        # type: (str, List[bool]) -> None
        select_node, leftover = apply_rule(select_rule, tokenize(select_query))
        self.assertFalse(leftover)

        exists = Exists(select_node)

        context = EvaluationContext(self.small_datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                                    EMPTY_NODE)
        dataframe = exists.evaluate(context)

        self.assertEqual(list(dataframe.series), result)

    def test_exists_reference_outer(self):
        datasets = {
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1], [4]], columns=['a']),
                        types=[BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[4], [2]], columns=['b']),
                        types=[BQScalarType.INTEGER]
                    ),
                }
            }
        }
        select_query = "select a from `my_project.my_dataset.my_table` where " \
                       "my_table.a = my_table2.b"
        select_node, leftover = apply_rule(select_rule, tokenize(select_query))
        self.assertFalse(leftover)

        exists = Exists(select_node)

        context = EvaluationContext(datasets)
        context.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                                    EMPTY_NODE)
        dataframe = exists.evaluate(context)

        self.assertEqual(list(dataframe.series), [True, False])

    def test_exists_index(self):
        datasets = {
            'my_project': {
                'my_dataset': {
                    'bool_table': TypedDataFrame(
                        pd.DataFrame([[True], [False]], columns=['a']),
                        types=[BQScalarType.BOOLEAN]
                    )
                }
            }
        }
        select_query = 'select a = exists(select 1) from `my_project.my_dataset.bool_table`'
        select_node, leftover = apply_rule(select_rule, tokenize(select_query))
        self.assertFalse(leftover)

        result, unused_table_name = select_node.get_dataframe(datasets)

        self.assertEqual(result.to_list_of_lists(), [[True], [False]])

    @data(
        ('DAYOFWEEK', 3),
        ('DAY', 9),
        ('DAYOFYEAR', 129),
        ('WEEK', 19),
        ('ISOWEEK', 19),
        ('MONTH', 5),
        ('QUARTER', 2),
        ('YEAR', 2019),
        ('ISOYEAR', 2019),
    )
    @unpack
    def test_extract(self, part, result):
        # type: (str, int) -> None
        extract = Extract(part, Value(pd.Timestamp('2019-05-09'), BQScalarType.TIMESTAMP))

        dataframe = extract.evaluate(EMPTY_CONTEXT)
        self.assertEqual(list(dataframe.series), [result])

    def test_extract_unimplemented(self):
        extract = Extract('WEEK(TUESDAY)',
                          Value(pd.Timestamp('2019-05-09'), BQScalarType.TIMESTAMP))

        with self.assertRaisesRegexp(NotImplementedError, 'WEEK\(TUESDAY\) not implemented'):
            extract.evaluate(EMPTY_CONTEXT)


if __name__ == '__main__':
    unittest.main()
