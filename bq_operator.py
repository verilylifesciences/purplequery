# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Rules for tokenizing and parsing binary operators.

This module exports two public identifiers.

BINARY_OPERATOR_PATTERN is a regular expression suitable for matching punctuation operators.

binary_operator_expression_rule is a function returning a recursive descent rule that, given a rule
for matching subexpressions, matches an expression where those subexpressions are optionally
separated by binary operators.  The result is parsed according to BigQuery precedence rules and
returns an abstract syntax tree node that evaluates the corresponding expression.
"""

import re
from typing import Callable, List, NamedTuple, Optional, Union  # noqa: F401

from binary_expression import BinaryExpression
from bq_abstract_syntax_tree import (AppliedRuleOutputType, EvaluatableNode,  # noqa: F401
                                     EvaluationContext, RuleType)
from bq_binary_operators import BINARY_OPERATOR_INFO
from query_helper import separated_sequence

# This pattern is used by the tokenizer to recognize operators named by punctuation.
# We start with the longest first so that matching < doesn't preclude matching << or <=.
BINARY_OPERATOR_PATTERN = '|'.join(
    sorted([re.escape(op) for op in BINARY_OPERATOR_INFO if not op.isalpha()],
           key=len, reverse=True))


def _reparse_binary_expression(unparsed_sequence):
    # type: (List[Union[str, EvaluatableNode]]) -> EvaluatableNode
    """Reparses a sequence of nodes and operators by operator precedence.

    Because a recursive descent parser can't represent left recursive grammars
    naturally (it would get caught in infinite recursion), we parse expressions
    using iteration rather than recursion.  That, however, fails to capture
    operator precedence, so 1 + 2 * 3 would be 9 instead of 6.  So, we take the
    output of the iterative grammar rule below, and re-parse it into a binary
    parse tree with the proper operator precedence.

    This is a simple shift-reduce parser.  We have a sequence of as-yet-unseen nodes
    (either parse tree nodes or operator strings separating them), and a stack of "consumed" nodes.

    At each stage, we consider the next node in the sequence and the nodes on the stack, and choose
    from two actions:
       shift: move one node from the sequence to the top of the stack
       reduce: take the top three nodes on the stack, wrap them in a BinaryExpression node,
           and put that node back on top of the stack.

    Args:
       unparsed_sequence: An alternating sequence of AST nodes and operator strings.
    Returns:
       A single AST node.
    """
    if len(unparsed_sequence) % 2 == 0:
        raise ValueError("Sequence must be of odd length: {!r}".format(unparsed_sequence))
    for operator_str in unparsed_sequence[1::2]:
        if operator_str not in BINARY_OPERATOR_INFO:
            raise ValueError("Unknown operator string {}".format(operator_str))
    for node in unparsed_sequence[::2]:
        if not isinstance(node, EvaluatableNode):
            raise ValueError("Sequence must alternate node, operator, node, ...: {!r} is not a node"
                             .format(node))

    parse_stack = []  # type: List[Union[str, EvaluatableNode]]

    # We're done when unparsed_sequence is empty and there's one thing left on the parse_stack
    while unparsed_sequence or len(parse_stack) != 1:
        if (
            # If there are no more nodes to consume, we have nothing to do but reduce.
            not unparsed_sequence
            # Otherwise, if there are at least three nodes on the parse_stack (node, operator, node)
            or (len(parse_stack) >= 3
                # And the next character is an operator, not an AST node
                and isinstance(unparsed_sequence[0], str) and
                # And the next character doesn't bind more tightly than what's on the parse_stack
                (BINARY_OPERATOR_INFO[parse_stack[-2]].precedence
                 <= BINARY_OPERATOR_INFO[unparsed_sequence[0]].precedence))):
            # then reduce!
            right_expression = parse_stack.pop()
            operator_str = parse_stack.pop()
            left_expression = parse_stack.pop()
            parse_stack.append(BinaryExpression(left_expression, operator_str, right_expression))
        else:
            # Otherwise, shift!
            parse_stack.append(unparsed_sequence.pop(0))
    # The loop has exited, therefore unparsed_sequence is empty and parse_stack has 1 element left.
    return parse_stack[0]


def binary_operator_expression_rule(subexpression_rule):
    # type: (RuleType) -> Callable[[List[str]], AppliedRuleOutputType]
    """Returns a rule for parsing binary expressions given a rule that the operators separate.

    Args:
        subexpression_rule: A grammar rule for some kind of expressions, i.e. a function
            taking tokens and returning a parsed abstract syntax tree node plus leftover tokens.
    Returns:
        A grammar rule that parses expressions where one or more subexpressions (as recognized by
            the rule passed in) are separated by binary operators, parsed according to operator
            precedence.
    """
    return separated_sequence(subexpression_rule,
                              sorted(BINARY_OPERATOR_INFO.keys(), key=len, reverse=True),
                              wrapper=_reparse_binary_expression,
                              keep_separator=True)
