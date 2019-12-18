# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest

from purplequery.tokenizer import remove_comments, tokenize


class TokenizerTest(unittest.TestCase):

    def test_remove_comments(self):
        # type: () -> None
        """Tests removal of comments"""
        self.assertEqual(
            remove_comments(
                'foo # end-of-line-with-pound\n'
                '# successive lines\n'
                '# of nothing but comments\n'
                'more and -- end-of-line-with --\n'
                '-- successive lines\n'
                '-- of nothing but comments\n'
                'with /* comments in the middle */ removed\n'
                'even /* comments *//* that *//* abut */ removed\n'
                'but /* does not end * / yet * still not */ now\n'),
            ('foo \n'
             '\n'
             '\n'
             'more and \n'
             '\n'
             '\n'
             'with  removed\n'
             'even  removed\n'
             'but  now\n'))

    def test_general_tokenization(self):
        # type: () -> None
        """Tests tokenization of numbers, strings, variables, comments, lists."""
        self.assertEqual(
            tokenize(
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
        # type: () -> None
        """Tests tokenization of functions, parentheses, comments."""
        self.assertEqual(
            tokenize("SELECT IF(x, '','concat')"),
            ['SELECT', 'IF', '(', 'x', ',', "''", ',', "'concat'", ')'])

    def test_like_order_by(self):
        # type: () -> None
        """Tests tokenization of operators like WHERE, LIKE, ORDER BY."""
        self.assertEqual(
            tokenize("SELECT * FROM mytable WHERE field1 LIKE 'a%' ORDER BY field2"),
            ['SELECT', '*', 'FROM', 'mytable', 'WHERE', 'field1', 'LIKE', "'a%'",
             'ORDER', 'BY', 'field2'])

    def test_join(self):
        # type: () -> None
        """Tests tokenization of syntax related to JOIN."""
        self.assertEqual(
            tokenize(
                '''SELECT * FROM (SELECT field1, field2 FROM table1) AS t1
                   JOIN (SELECT field3, field4 FROM table2) AS t2 ON
                   t1.field1=t2.field2
                '''),
            ['SELECT', '*', 'FROM', '(', 'SELECT', 'field1', ',', 'field2', 'FROM', 'table1', ')',
             'AS', 't1', 'JOIN', '(', 'SELECT', 'field3', ',', 'field4', 'FROM', 'table2', ')',
             'AS', 't2', 'ON', 't1', '.', 'field1', '=', 't2', '.', 'field2'])

    def test_match_order(self):
        # type: () -> None
        """Tests tokenization of operators that share characters."""
        self.assertEqual(
            tokenize("SELECT f1 <> 3, f2 << 2, f3 < 5 AND f3 > -1 FROM t"),
            ['SELECT', 'f1', '<>', '3', ',', 'f2', '<<', '2', ',', 'f3', '<', '5', 'AND',
             'f3', '>', '-', '1', 'FROM', 't'])

    def test_negatives(self):
        # type: () -> None
        """Tests negative and positive numbers."""
        self.assertEqual(
            tokenize("SELECT -1.23e1, -1e1, -.23e2, 4.0e-7, +5, +3. FROM t"),
            ['SELECT', '-', '1.23e1', ',', '-', '1e1', ',', '-', '.23e2', ',', '4.0e-7',
             ',', '+', '5', ',', '+', '3.', 'FROM', 't'])


if __name__ == '__main__':
    unittest.main()
