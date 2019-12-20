# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Grammar for BigQuery statements."""

from .grammar import expression, query_expression
from .query_helper import separated_sequence
from .statements import CreateTable, CreateView
from .terminals import grammar_literal, identifier
from .type_grammar import bigquery_type

statement = [
    (CreateTable,
     [grammar_literal('CREATE', 'TABLE', 'IF', 'NOT', 'EXISTS'),
      grammar_literal('CREATE', 'TABLE'),
      grammar_literal('CREATE', 'OR', 'REPLACE', 'TABLE')],
     separated_sequence(identifier, '.'),
     [('(', separated_sequence((identifier, bigquery_type), ','), ')'),
      None],
     # PARTITION BY not implemented
     # CLUSTER BY not implemented
     [('OPTIONS', '(',
       [separated_sequence((identifier, '=', expression), ','), None],
       ')'),
      None],
     [('AS', '(', query_expression, ')'),
      None],
     ),
    (CreateView,
     [grammar_literal('CREATE', 'VIEW', 'IF', 'NOT', 'EXISTS'),
      grammar_literal('CREATE', 'VIEW'),
      grammar_literal('CREATE', 'OR', 'REPLACE', 'VIEW')],
     separated_sequence(identifier, '.'),
     [('OPTIONS', '(',
       [separated_sequence((identifier, '=', expression), ','), None],
       ')'),
      None],
     'AS',
     '(', query_expression, ')'
     ),
]


bigquery_statement = ([statement, query_expression],
                      [';', None])
