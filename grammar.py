# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''An implementation of the Google BigQuery Standard SQL syntax.

https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax

Differences from this grammar occur for two reasons.
- The original grammar is left recursive, and this is a recursive descent
  parser, so we need to juggle things to make that work.

In addition, some expressions have their own unique grammar; these are not necessarily implemented.

In a recursive descent parser: each grammar rule corresponds to a function.  Each function
takes a list of tokens, and returns a pair, of the identified node and the remaining unparsed
tokens, or None and all the tokens if the rule doesn't match.

To simplify the grammar, we allow rules to be specified in a few ways that are not Python functions.

A rule that is a literal string means a rule matching that string.

A rule that is a tuple of rules means all of the contained rules must match in order.

A rule that is a list of rules means exactly one of the rules must match.

A rule that is None matches zero tokens; it's a placeholder to allow, with lists, for rules to
be optional.

This logic is implemented by the apply_rule function.  All rules must use this function to apply
a rule to input tokens.

Given this, we still implement some rules as functions.  This is for one of two reasons.
- The function uses parsing logic other than recursive descent; data_source is an example.
- The rule depends on other rules that also depend on this rule, so we define
  one as a function to allow the mutual recursion (since Python doesn't have let rec).


Note that this file is a mix of 'constants' and functions.  All grammar rules are notionally
functions because this is a recursive descent parser, but the apply_rule helper function lets us
write simpler rules as constants (tuples, lists, etc).  More complex rules are defined as actual
functions.  Therefore constants are formatted the same as functions here, because they may become
become them as they grow in complexity.
'''

from typing import List, cast  # noqa: F401

from bq_abstract_syntax_tree import EMPTY_NODE, Field
from bq_operator import binary_operator_expression_rule
from dataframe_node import QueryExpression, Select, SetOperation, TableReference
from evaluatable_node import (Array, Array_agg, Case, Cast, Count, Exists, Extract, FunctionCall,
                              If, InCheck, Not, NullCheck, Selector, StarSelector, UnaryNegation)
from join import DataSource, FromItemType, Join
from query_helper import AppliedRuleOutputType  # noqa: F401
from query_helper import apply_rule, separated_sequence
from terminals import grammar_literal, identifier, literal


def field(tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    '''A field is a column reference in the format TableName.ColumnName or just ColumnName.

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the Field node representing the result of applying the rule
        to the tokens, and the remaining unmatched tokens
    '''
    # TableName.ColumnName rule: (identifier, '.', identifier)
    # ColumnName rule: identifier
    field_path, new_tokens = apply_rule(
        [(identifier, '.', identifier), identifier],
        tokens)

    # Field initializer always expects a tuple, but identifier() will return a
    # string
    if not field_path:
        return None, tokens
    elif not isinstance(field_path, tuple):
        field_path = (field_path,)
    return Field(field_path), new_tokens


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
        return wrapper(*result), new_tokens
    return wrapped_rule


def function_call(tokens):
    '''Grammar rule matching function calls.

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the function call node representing the result of applying the rule
        to the tokens, and the remaining unmatched tokens, or None and all the tokens if
        the rule does not match.
    '''
    matched_function, new_tokens = apply_rule(
            (identifier, '(', [separated_sequence(expression, ','), None], ')', [over_clause,
                                                                                 None]),
            tokens)
    if matched_function is None:
        return None, tokens
    return FunctionCall.create(*matched_function), new_tokens


def core_expression(tokens):
    """Grammar rule for a core set of expressions that can be nested inside other expressions.

    The current set of handled expressions are:
    - Array
    - Array_agg
    - Count
    - Function call
    - A field (column)
    - Case
    - A literal (number, string, etc)
    - If
    - Cast
    - Exists
    - Extract
    - Another expression nested in parentheses
    - Not
    - Unary negation
    """
    return apply_rule(
        [
            # COUNT(*), COUNT(DISTINCT expression), COUNT(expression)
            wrap(Count.create_count_function_call,
                 ('COUNT', '(', ['*', (['DISTINCT', None], expression)], ')', [over_clause, None])),

            wrap(Array_agg.create_function_call,
                 ('ARRAY_AGG', '(', ['DISTINCT', None], expression,
                  [(['IGNORE', 'RESPECT'], 'NULLS'), None],
                  [(grammar_literal('ORDER', 'BY'),
                    separated_sequence(identifier, ['ASC', 'DESC', None], ',')),
                   None],
                  [('LIMIT', literal), None],
                  ')',
                  [over_clause, None])),

            wrap(FunctionCall.create,
                 (identifier, '(', [separated_sequence(expression, ','), None], ')', [over_clause,
                                                                                      None])),
            function_call,

            field,

            (Case, [expression, None], 'WHEN',
                separated_sequence((expression, 'THEN', expression), 'WHEN'),
                [('ELSE', expression), None], 'END'),

            literal,

            (If, '(', expression, ',', expression, ',', expression, ')'),

            (Cast, '(', expression, 'AS', identifier, ')'),

            (Exists, '(', query_expression, ')'),

            (Array, [('ARRAY', '<', identifier, '>'), None],
             '[', [separated_sequence(expression, ','), None], ']'),

            (Extract, '(', identifier, 'FROM', expression, ')'),

            ('(', expression, ')'),

            (Not, expression),

            (UnaryNegation, '-', expression),
        ],
        tokens)


def post_expression(tokens):
    """Grammar rule for expressions that occur only after a core expression.

    For example:
    <core_expression> IS NULL
    <core_expression> IN (a, b, c)

    If the query has none of these, it can still match a plain `core_expression`,
    the last item in the list.  [Currently this is the only thing implemented.]
    """
    return apply_rule(
        [
            (NullCheck, core_expression, [grammar_literal('IS', 'NULL'),
                                          grammar_literal('IS', 'NOT', 'NULL')]),
            (InCheck, core_expression, ['IN', grammar_literal('NOT', 'IN')],
             '(', separated_sequence(expression, ','), ')'),
            core_expression,
        ],
        tokens)


# Grammar rule for expressions that can be nested inside other expressions.
# It can be a plain core_expression, a post_expression (core_expression with
# additional content at the end), or a sequence of post_expressions separated by
# binary operators.
expression = binary_operator_expression_rule(post_expression)

# Grammar rule for a clause added to analytic expressions to specify what window the function
# is evaluated over.
#
# See full syntax here:
# https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts#analytic-function-syntax
# Window identifiers and window frames are not supported.
over_clause = ('OVER',
               '(',
               [('PARTITION', 'BY', separated_sequence(expression, ',')), None],
               [('ORDER', 'BY', separated_sequence((expression, ['ASC', 'DESC', None]), ',')),
                None],
               ')')


# [Optional] "ORDER BY" followed by some number of expressions to order by, and a sort direction
maybe_order_by = [('ORDER', 'BY',
                   separated_sequence((expression, ['ASC', 'DESC', None]), ',')),
                  None]

# [Optional] "LIMIT" followed by the number of rows to return, and an optional
# "OFFSET" to indicate which row to start at
maybe_limit = [('LIMIT', literal, [('OFFSET', literal), None]), None]


# A set operator, combining the results of separate queries.
set_operator = [
    grammar_literal('UNION', 'ALL'),
    grammar_literal('UNION', 'DISTINCT'),
    grammar_literal('INTERSECT', 'DISTINCT'),
    grammar_literal('EXCEPT', 'DISTINCT')]


def query_expression(tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    '''This is the highest-level grammar method.  It is called by query.execute_query().

    The "raw" rule syntax is supposedly this:
    https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax

    query_expression = ([with_expression, None],
                        [select,
                         ('(', query_expression, ')'),
                         (query_expression, set_operator, query_expression)],
                        maybe_order_by,
                        maybe_limit)

    However, (a) this is left-recursive, so we would need to juggle things to avoid
    infinite recursion, and (b) this permits things like
    WITH query1 as (SELECT 1) WITH query2 as (SELECT 2) SELECT 3 UNION ALL SELECT 4
    that (1) seem kind of strange (two successive with clauses)
    and (2) give an error on prod BigQuery.

    So we use the following simpler syntax.

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the Abstract Syntax Tree nodes representing the result of
        applying the rule to the tokens, and the remaining unmatched tokens.
    '''
    core_query_expression = (
        QueryExpression,
        [with_expression, None],
        [select,
         ('(', query_expression, ')')],

        # order_by
        maybe_order_by,

        # limit
        maybe_limit)

    return apply_rule(
        [(SetOperation, core_query_expression, set_operator, query_expression),
         core_query_expression],
        tokens)


with_expression = ('WITH', separated_sequence((identifier, 'AS', '(', query_expression, ')'), ','))


def alias(tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    '''An optional alias to rename a field or table.

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the matched alias identifier (if any), and the remaining tokens.
    '''
    alias_node, new_tokens = apply_rule([
        (['AS', None], identifier),
        None
    ], tokens)

    if alias_node == EMPTY_NODE:
        return EMPTY_NODE, tokens

    if not (isinstance(alias_node, tuple) and len(alias_node) == 2):
        raise RuntimeError("Internal parse error: alias rule returned {!r}".format(alias_node))

    # The alias node will be a tuple of ('AS', new_name), so we get rid of the
    # 'AS' and just return the new name (alias identifier)
    as_token, alias_identifier = alias_node
    return alias_identifier, new_tokens


def select(tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    '''Grammar rule matching a select clause.

    This rule is adapted from here:
    https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#select-list

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the Abstract Syntax Tree nodes representing the result of
        applying the rule to the tokens, and the remaining unmatched tokens.
    '''
    return apply_rule(
        (Select,

         # modifier
         ['ALL', 'DISTINCT', None],

         # fields
         separated_sequence(

              # The selector either contains a * or is just an expression
              [
                  # If selector's expression includes a '*', it can also include
                  # exception and replacement
                  (
                      StarSelector,

                      # expression - can be just * or field.* or table.field.*, etc
                      [(expression, '.'), None],

                      '*',

                      # exception
                      None,

                      # replacement
                      None
                  ),

                  # The selector does not include a * and therefore cannot have
                  # an exception or replacement
                  (Selector, expression, alias),
              ],

              # separator for separated_sequence()
              ','
         ),

         # from
         [('FROM', data_source), None],

         # where
         [('WHERE', expression), None],

         # group by
         [('GROUP', 'BY', separated_sequence([field, literal], ',')), None],

         # having
         [('HAVING', expression), None],
         ),
        tokens)


# Expression following "FROM": a table or another query, followed optionally by an AS alias
# Examples: "SomeTable", "(SELECT a from SomeTable)", "SomeTable AS t"
from_item = ([
    (TableReference, separated_sequence(identifier, '.')),
    ('(', query_expression, ')')
], alias)


join_type = ['INNER',
             'CROSS',
             grammar_literal('FULL', 'OUTER'), 'FULL',
             grammar_literal('LEFT', 'OUTER'), 'LEFT',
             grammar_literal('RIGHT', 'OUTER'), 'RIGHT',
             None]


def data_source(orig_tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    '''Includes the initial FROM expression as well as any following JOINs.

    This describes everything that comes after a FROM, essentially in the form:
    from_item JOIN from_item ON on_expression JOIN from_item ON on_expression JOIN ...

    The first from_item is called first_from and is required.

    After that is any number of repetititions of ('JOIN', from_item, (ON, on_expression)),
    and is parsed into an array (possibly empty) called joins.

    Args:
        orig_tokens: Parts of the user's query (split by spaces into tokens)
        that are not yet parsed
    Returns:
        A tuple of the Abstract Syntax Tree nodes representing the result of
        applying the rule to the tokens, and the remaining unmatched tokens.
    '''
    first_from, tokens = apply_rule(from_item, orig_tokens)
    if not first_from:
        return None, tokens
    joins = []
    while True:
        next_join, tokens = apply_rule([(',', from_item),  # shorthand for cross-join
                                        (join_type,
                                         'JOIN',
                                         from_item,
                                         [('ON', expression),
                                          ('USING', '(', separated_sequence(identifier, ','), ')'),
                                          None])],
                                       tokens)
        if next_join:
            # This case is triggered by the shorthand cross-join above where the table to be joined
            # is just specified separated by a comma, with no join type or condition specified.
            if not isinstance(next_join, tuple):
                raise RuntimeError("Internal error; join rule above must result in tuple not {}"
                                   .format(next_join))
            if len(next_join) == 2:
                joins.append(Join('CROSS', cast(FromItemType, next_join), EMPTY_NODE))
            else:
                joins.append(Join(*next_join))
        else:
            break
    return DataSource(cast(FromItemType, first_from), joins), tokens
