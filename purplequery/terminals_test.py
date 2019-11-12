# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest

from purplequery.bq_types import BQScalarType
from purplequery.evaluatable_node import Value
from purplequery.terminals import grammar_literal, identifier, literal


class TerminalsTest(unittest.TestCase):

    def test_identifier_simple(self):
        '''Check simple identifier match'''
        self.assertEqual(identifier(['abc']), ('abc', []))

    def test_identifier_fail_because_reserved_word(self):
        '''Don't match because 'select' is a reserved word'''
        self.assertEqual(
            identifier(['select', 'abc', 'from', 't']),
            (None, ['select', 'abc', 'from', 't']))

    def test_identifier_backticks(self):
        '''Remove backticks when matching'''
        self.assertEqual(
            identifier(['`abc`', 'def']),
            ('abc', ['def']))

    def test_identifier_backticks_reserved_word(self):
        '''Match even though SELECT is a reserved word, because it's in backticks'''
        self.assertEqual(
            identifier(['`SELECT`', 'def']),
            ('SELECT', ['def']))

    def test_literals_float(self):
        '''Check floats'''
        self.assertEqual(
            literal(['1.23']),
            (Value(1.23, BQScalarType.FLOAT), []))

    def test_literals_integer(self):
        '''Check integers'''
        self.assertEqual(
            literal(['11']),
            (Value(11, BQScalarType.INTEGER), []))

    def test_literals_string(self):
        '''Check strings'''
        self.assertEqual(
            literal(['"something"']),
            (Value('something', BQScalarType.STRING), []))

    def test_literals_null(self):
        '''Check NULL'''
        self.assertEqual(
            literal(['NULL']),
            (Value(None, None), []))

    def test_literals_true(self):
        '''Check TRUE and also that remainer gets returned'''
        self.assertEqual(
            literal(['TRUE', 'abc']),
            (Value(True, BQScalarType.BOOLEAN), ['abc']))

    def test_literals_false(self):
        '''Check FALSE and also lower to upper conversion'''
        self.assertEqual(
            literal(['false']),
            (Value(False, BQScalarType.BOOLEAN), []))

    def test_literals_fail_because_identifier(self):
        '''Don't match because 'abc' is not a literal'''
        self.assertEqual(
            literal(['abc']),
            (None, ['abc']))

    def test_grammar_literal(self):
        '''Check multi-word reserved words'''
        self.assertEqual(
            grammar_literal('ORDER', 'BY')(['ORDER', 'BY', 'field']),
            ('ORDER_BY', ['field']))

    def test_grammar_literals_lower_to_upper(self):
        '''Match even though 'select' is in lowercase'''
        self.assertEqual(
            grammar_literal('select')(['select']),
            ('SELECT', []))

    def test_grammar_literals_fail(self):
        '''Don't match because the reserved word isn't the next token'''
        self.assertEqual(
            grammar_literal('SELECT')(['abc']),
            (None, ['abc']))


if __name__ == '__main__':
    unittest.main()
