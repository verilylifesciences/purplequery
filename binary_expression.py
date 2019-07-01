"""An evaluatable AST node for binary expressions.
Separated file to resolve circular imports.

Tested in bq_operator_test.py
"""

from typing import List, Sequence  # noqa: F401

from bq_abstract_syntax_tree import (EvaluatableNode, EvaluatableNodeWithChildren,  # noqa: F401
                                     EvaluationContext)
from bq_binary_operators import BINARY_OPERATOR_INFO
from bq_types import TypedSeries, implicitly_coerce


class BinaryExpression(EvaluatableNodeWithChildren):

    def __init__(self, left, operator_str, right):
        # type: (EvaluatableNode, str, EvaluatableNode) -> None
        self.children = [left, right]
        self.operator_info = BINARY_OPERATOR_INFO[operator_str]

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> BinaryExpression
        new_left, new_right = new_children
        return BinaryExpression(new_left, self.operator_info.operator, new_right)

    def strexpr(self):
        # type: () -> str
        """Returns a prefix-expression serialization for testing purposes."""
        left, right = self.children
        return '({} {} {})'.format(self.operator_info.operator,
                                   left.strexpr(),
                                   right.strexpr())

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        left_value, right_value = evaluated_children
        # We need to know the type of the result of the operation.  Some operators specify their
        # result type (e.g. comparators have a boolean output regardless of input types).  Some
        # operators keep the type of their inputs (e.g. multiplication keeps the input numerical
        # types); this is notated in the BINARY_OPERATOR_INFO table by specifying the result_type
        # as None.  The right thing to do if the types of the inputs aren't the same is to apply
        # specific logic depending on the input types: see here for the logic for arithmetic
        # operators:
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/operators#arithmetic_operators
        result_type = self.operator_info.result_type
        if not result_type:
            result_type = implicitly_coerce(left_value.type_, right_value.type_)
        return TypedSeries(self.operator_info.function(left_value.series, right_value.series),
                           result_type)
