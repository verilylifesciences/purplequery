# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''Helpers for writing the BigQuery SQL grammar concisely.

In a recursive descent parser, grammar rules are functions.  Instead of needing to write each rule
as a function, the helpers in this module allow us to be more concise.

The apply_rule function removes the requirement that all rules are functions, so long as all rules
use this function to apply a rule to input tokens.  The rules it allows are as follows:

A rule that is a literal string means a rule matching that string.

A rule that is a tuple of rules means all of the contained rules must match in order.

A rule that is a list of rules means exactly one of the rules must match.

A rule that is None matches zero tokens; it's a placeholder to allow, with lists, for rules to
be optional.


The separated_sequence function returns a grammar rule matching one or more occurrences of a rule,
separated by ocurrences of another rule.  Examples include expressions with binary operators, or
the arguments to a function.

The wrap function calls a specified function on the result of a successful rule application,
allowing construction of an object from the rule's output.
'''


from typing import Any, Callable, List, Optional, Tuple, Union  # noqa: F401

from bq_abstract_syntax_tree import (EMPTY_NODE, AbstractSyntaxTreeNode,  # noqa: F401
                                     AppliedRuleNode, AppliedRuleOutputType, RuleType)
from terminals import grammar_literal


def separated_sequence(rule,  # RuleType
                       separator_rule,  # RuleType
                       wrapper=tuple,  # Callable[[List[AbstractSyntaxTreeNode]], AbstractSyntaxTreeNode] # noqa: E501
                       keep_separator=False  # bool
                       ):
    # type: (...) -> Callable[[List[str]], AppliedRuleOutputType]
    """Return a grammar rule that is a sequence of `rule`s with a separator.

    The returned rule `result` is equivalent to

    result :: = rule separator_rule result | rule

    Args:
        rule: A grammar rule.
        separator_rule: A grammar rule, used as a separator.
        wrapper: An optional class; if provided, it will be called with the matching
           syntax tree nodes as arguments to create the return value.
        keep_separator: Nodes representing the separators are discarded unless this is True.

    Returns:
        A grammar rule, i.e. a function taking a list of tokens, and returning a result
        and all non-matching tokens.  That result is:

            None (if the rule doesn't match)
            An object of type class, constructed with all matching nodes (if cls is provided)
            node (if one token matches `rule` and the separator isn't a literal string)
            a tuple of nodes matching (rule, rule, ... rule) (if separator is a literal string)
            a tuple of nodes matching (rule, separator, rule, separator, ... rule) (otherwise)
    """
    def check_sequence(tokens):
        # type: (List[str]) -> AppliedRuleOutputType
        nodes = []
        # Initial value must be non-None but also have the correct type
        maybe_separator = AbstractSyntaxTreeNode()  # type: AppliedRuleNode
        tokens_after_separator = tokens

        while maybe_separator:
            # Try to match the given rule
            node, tokens = apply_rule(rule, tokens_after_separator)
            if not node:
                break
            nodes.append(node)

            # Try to match the given separator
            maybe_separator, tokens_after_separator = apply_rule(separator_rule, tokens)
            if maybe_separator and keep_separator:
                nodes.append(maybe_separator)
        if not nodes:
            return None, tokens
        return wrapper(nodes), tokens
    return check_sequence


def _apply_rule_str(rule, tokens):
    # type: (str, List[str]) -> AppliedRuleOutputType
    '''If the rule is a string, just check if the next token is that string.

    This usually comes up as part of a more complex rule.  For example:

    apply_rule('SELECT', ['SELECT', '*', 'FROM', 'SomeTable']) ->
        Match: 'SELECT'
        Rest of tokens: ['*', 'FROM', 'SomeTable']

    apply_rule('(', ['(', '1', '+', '2', ')'])
        Match: '('
        Rest of tokens: ['1', '+', '2', ')']
    '''
    return grammar_literal(rule)(tokens)


def _apply_rule_tuple(rule, tokens):
    # type: (Tuple[RuleType, ...], List[str]) -> AppliedRuleOutputType
    '''If the rule is a tuple, check that the next tokens correspond to each of
    the elements in the tuple (all subrules must match, in order).

    There is a common special case where the first element in the tuple will
    be an AbstractSyntaxTreeNode class.  In that case, this class gets
    initialized with the rest of the matching tokens.

    Examples:

    apply_rule(('ORDER', 'BY', separated_sequence(field, ',')), \
        ['ORDER', 'BY', 'FieldOne', ',', 'FieldTwo'])
        Match: ('FieldOne', 'FieldTwo')
        Rest of tokens: []

        This rule requires that the tokens start with the string 'ORDER',
        followed by 'BY', and then a sequence of fields separated by commas.

        The 'ORDER', 'BY', and comma in this example are matched, but not
        included in the return value.

    apply_rule((If, '(', expression, ',', expression, ',', expression, ')'), \
        ['IF', '(', 'a', '>', '0', ',', 'b', ',', 'c', ')'])
        Match: instance of `If` class with its fields set to:
            condition: a > 0
            then: b
            else_: c
        Rest of tokens: []

        In this case `If` is a class that is a subtype of
        AbstractSyntaxTreeNode.  This class has a literal defined ('IF').

        So, when trying to match tokens, this rule would look for the string
        'IF', followed by parentheses, which must encapsulate 3 expressions
        separated by commas.

        After matching all parts of the rule, an `If` instance would be
        instantiated with the 3 expressions getting passed in as the 3
        arguements to `__init__()`: `condition`, `then`, and `else_`.
    '''
    nodes = []
    orig_tokens = tokens
    node_class = None
    for subrule in rule:
        # If the rule is an AST Node, try to match it to its literal marker
        # (eg, "SELECT" for a Select AST Node).  If it doesn't have one,
        # don't try to match it because there's nothing to match.
        if isinstance(subrule, type(AbstractSyntaxTreeNode)) and \
                issubclass(subrule, AbstractSyntaxTreeNode):
            node_class = subrule
            if subrule.literal():
                subrule = subrule.literal()
            else:
                continue

        node, tokens = apply_rule(subrule, tokens)

        # If at least one failed to match, this rule does not match
        if not node:
            return None, orig_tokens

        # Don't include string constants from the grammar in the AST
        if not isinstance(subrule, str):
            nodes.append(node)

    # Reformat the results because the return value cannot be a list (see
    # AppliedRuleOutputType)
    output = None
    if node_class:
        # If an AST Node class type was given, initialize the AST Node
        output = node_class(*nodes)
    elif len(nodes) == 1:
        # A list with only one element is simplified to just that element
        output = nodes[0]
    else:
        # A list with multiple elements should really be a tuple
        output = tuple(nodes)
    return output, tokens


def _apply_rule_list(rule, tokens):
    # type: (List[RuleType], List[str]) -> AppliedRuleOutputType
    '''A rule that is a list represents a set of possible alternatives, which
    are checked in turn against the next token.

    This short-circuits after the finding the first subrule that matches.

    Example:
    apply_rule(['IN', grammar_literal('NOT', 'IN')], ['IN'])
        Match: 'IN'
        Rest of tokens: []

    apply_rule(['IN', grammar_literal('NOT', 'IN')], ['NOT', 'IN'])
        Match: 'NOT_IN'
        Rest of tokens: []

    Frequently used with None (see below) to indicate optional elements.
    '''
    for subrule in rule:
        node, new_tokens = apply_rule(subrule, tokens)
        if node:
            return node, new_tokens
    return None, tokens


def _apply_rule_none(rule, tokens):
    # type: (None, List[str]) -> AppliedRuleOutputType
    '''"None" means this is an epsilon rule - it matches no input tokens and
    always succeeds.  This is mainly used as an element in a list to indicate
    that the list rule is optional.

    Example:
    apply_rule(['ALL', 'DISTINCT', None]), ['ALL'])
        Match: 'ALL'
        Rest of tokens: []

    apply_rule(['ALL', 'DISTINCT', None]), ['DISTINCT'])
        Match: 'DISTINCT'
        Rest of tokens: []

    apply_rule(['ALL', 'DISTINCT', None]), ['*'])
        Match: EMPTY_NODE
        Rest of tokens: ['*']

    The "None" makes the entire list subrule optional.
    '''
    return EMPTY_NODE, tokens


def _apply_rule_method(rule, tokens):
    # type: (Callable[[List[str]], AppliedRuleOutputType], List[str]) -> AppliedRuleOutputType
    '''If the rule is a method, apply the method to the tokens.

    Example:
    apply_rule(('LIMIT', literal), ['LIMIT', '10'])
        Match: ('LIMIT', Value(value=10, type_=BQScalarType.INTEGER))
        Rest of tokens: []

        `literal()` is a method defined in terminals.py that matches a
        number, string, boolean, or null.
    '''
    return rule(tokens)


def apply_rule(rule, tokens):
    # type: (RuleType, List[str]) -> AppliedRuleOutputType
    """Apply the given rule to tokens if possible.

    Args:
        rule: The object (string, tuple, list, method, None) that represents the rule.
        tokens: A list of tokens.
    Returns:
        A tuple of the Abstract Syntax Tree nodes representing the result of applying the rule
        to the tokens, and the remaining unmatched tokens.
    """

    if isinstance(rule, str):
        return _apply_rule_str(rule, tokens)
    elif isinstance(rule, tuple):
        return _apply_rule_tuple(rule, tokens)
    elif isinstance(rule, list):
        return _apply_rule_list(rule, tokens)
    elif rule is None:
        return _apply_rule_none(rule, tokens)
    else:
        return _apply_rule_method(rule, tokens)


def wrap(wrapper, rule):
    '''Returns a grammar rule that, if `rule' matches, wraps the result by calling `wrapper'

    Args:
        wrapper: A function to wrap the rule's result.
        rule: Any grammar rule
    Returns:
        If the rule matches, the result of calling wrapper on the rule's result, plus leftover
        unmatched tokens.  Otherwise, returns None and all the tokens.
    '''
    def wrapped_rule(tokens):
        # type: (List[str]) -> AppliedRuleOutputType
        result, new_tokens = apply_rule(rule, tokens)
        if result is None:
            return None, tokens
        if (isinstance(rule, tuple)
                and len([subrule for subrule in rule if not isinstance(subrule, str)]) > 1):
            return wrapper(*result), new_tokens
        else:
            return wrapper(result), new_tokens
    return wrapped_rule
