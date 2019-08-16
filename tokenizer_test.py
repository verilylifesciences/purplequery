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

import tokenizer


class TokenizerTest(unittest.TestCase):

    def test_general_tokenization(self):
        """Test tokenization of numbers, strings, variables, comments, lists."""
        self.assertEqual(
            tokenizer.tokenize(
                '''SELECT 34.25, .92, 14., 78.23e-24, 12+23* hello as foo_bar0234, "nope", -- first
                   ARRAY<INT64>[1,2]
                   FROM `myproject.mydataset.mytable` -- more stuff
                   WHERE goodbye <= False
                '''),
            ['SELECT', '34.25', ',', '.92', ',', '14.', ',', '78.23e-24', ',', '12', '+', '23', '*',
             'hello', 'as', 'foo_bar0234', ',', '"nope"', ',', 'ARRAY', '<', 'INT64', '>', '[', '1',
             ',', '2', ']',
             'FROM', '`myproject.mydataset.mytable`',
             'WHERE', 'goodbye', '<=', 'False'])

    def test_function(self):
        """Test tokenization of functions, parentheses, comments."""
        self.assertEqual(
            tokenizer.tokenize("SELECT IF(x, '','concat')"),
            ['SELECT', 'IF', '(', 'x', ',', "''", ',', "'concat'", ')'])

    def test_like_order_by(self):
        """Test tokenization of operators like WHERE, LIKE, ORDER BY."""
        self.assertEqual(
            tokenizer.tokenize("SELECT * FROM mytable WHERE field1 LIKE 'a%' ORDER BY field2"),
            ['SELECT', '*', 'FROM', 'mytable', 'WHERE', 'field1', 'LIKE', "'a%'",
             'ORDER', 'BY', 'field2'])

    def test_join(self):
        """Test tokenization of syntax related to JOIN."""
        self.assertEqual(
            tokenizer.tokenize(
                '''SELECT * FROM (SELECT field1, field2 FROM table1) AS t1
                   JOIN (SELECT field3, field4 FROM table2) AS t2 ON
                   t1.field1=t2.field2
                '''),
            ['SELECT', '*', 'FROM', '(', 'SELECT', 'field1', ',', 'field2', 'FROM', 'table1', ')',
             'AS', 't1', 'JOIN', '(', 'SELECT', 'field3', ',', 'field4', 'FROM', 'table2', ')',
             'AS', 't2', 'ON', 't1', '.', 'field1', '=', 't2', '.', 'field2'])

    def test_match_order(self):
        """Test tokenization of operators that share characters."""
        self.assertEqual(
            tokenizer.tokenize("SELECT f1 <> 3, f2 << 2, f3 < 5 AND f3 > -1 FROM t"),
            ['SELECT', 'f1', '<>', '3', ',', 'f2', '<<', '2', ',', 'f3', '<', '5', 'AND',
             'f3', '>', '-', '1', 'FROM', 't'])

    def test_negatives(self):
        """Test negative and positive numbers."""
        self.assertEqual(
            tokenizer.tokenize("SELECT -1.23e1, -1e1, -.23e2, 4.0e-7, +5, +3. FROM t"),
            ['SELECT', '-', '1.23e1', ',', '-', '1e1', ',', '-', '.23e2', ',', '4.0e-7',
             ',', '+', '5', ',', '+', '3.', 'FROM', 't'])


if __name__ == '__main__':
    unittest.main()
