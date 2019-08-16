# Copyright 2019 Verily Life Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import terminals
from bq_types import BQScalarType
from evaluatable_node import Value


class TerminalsTest(unittest.TestCase):

    def test_identifier_simple(self):
        '''Check simple identifier match'''
        self.assertEqual(
            terminals.identifier(['abc']),
            ('abc', []))

    def test_identifier_fail_because_reserved_word(self):
        '''Don't match because 'select' is a reserved word'''
        self.assertEqual(
            terminals.identifier(['select', 'abc', 'from', 't']),
            (None, ['select', 'abc', 'from', 't']))

    def test_identifier_backticks(self):
        '''Remove backticks when matching'''
        self.assertEqual(
            terminals.identifier(['`abc`', 'def']),
            ('abc', ['def']))

    def test_identifier_backticks_reserved_word(self):
        '''Match even though SELECT is a reserved word, because it's in backticks'''
        self.assertEqual(
            terminals.identifier(['`SELECT`', 'def']),
            ('SELECT', ['def']))

    def test_literals_float(self):
        '''Check floats'''
        self.assertEqual(
            terminals.literal(['1.23']),
            (Value(1.23, BQScalarType.FLOAT), []))

    def test_literals_integer(self):
        '''Check integers'''
        self.assertEqual(
            terminals.literal(['11']),
            (Value(11, BQScalarType.INTEGER), []))

    def test_literals_string(self):
        '''Check strings'''
        self.assertEqual(
            terminals.literal(['"something"']),
            (Value('something', BQScalarType.STRING), []))

    def test_literals_null(self):
        '''Check NULL'''
        self.assertEqual(
            terminals.literal(['NULL']),
            (Value(None, None), []))

    def test_literals_true(self):
        '''Check TRUE and also that remainer gets returned'''
        self.assertEqual(
            terminals.literal(['TRUE', 'abc']),
            (Value(True, BQScalarType.BOOLEAN), ['abc']))

    def test_literals_false(self):
        '''Check FALSE and also lower to upper conversion'''
        self.assertEqual(
            terminals.literal(['false']),
            (Value(False, BQScalarType.BOOLEAN), []))

    def test_literals_fail_because_identifier(self):
        '''Don't match because 'abc' is not a literal'''
        self.assertEqual(
            terminals.literal(['abc']),
            (None, ['abc']))

    def test_grammar_literal(self):
        '''Check multi-word reserved words'''
        self.assertEqual(
            terminals.grammar_literal('ORDER', 'BY')(['ORDER', 'BY', 'field']),
            ('ORDER_BY', ['field']))

    def test_grammar_literals_lower_to_upper(self):
        '''Match even though 'select' is in lowercase'''
        self.assertEqual(
            terminals.grammar_literal('select')(['select']),
            ('SELECT', []))

    def test_grammar_literals_fail(self):
        '''Don't match because the reserved word isn't the next token'''
        self.assertEqual(
            terminals.grammar_literal('SELECT')(['abc']),
            (None, ['abc']))


if __name__ == '__main__':
    unittest.main()
