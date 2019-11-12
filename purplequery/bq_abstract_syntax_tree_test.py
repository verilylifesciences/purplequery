# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from typing import List, Tuple, Type, Union  # noqa: F401

import pandas as pd
from ddt import data, ddt, unpack

from purplequery.bq_abstract_syntax_tree import (EMPTY_NODE, AbstractSyntaxTreeNode,  # noqa: F401
                                                 DatasetTableContext, EvaluationContext,
                                                 TableContext, _EmptyNode)
from purplequery.bq_types import BQScalarType, TypedDataFrame
from purplequery.dataframe_node import TableReference


@ddt
class BQAbstractSyntaxTreeTest(unittest.TestCase):

    def setUp(self):
        # type: () -> None
        self.table_context = DatasetTableContext({
            'my_project': {
                'my_dataset': {
                    'my_table': TypedDataFrame(
                        pd.DataFrame([[1], [2]], columns=['a']),
                        types=[BQScalarType.INTEGER]
                    ),
                    'my_table2': TypedDataFrame(
                        pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']),
                        types=[BQScalarType.INTEGER, BQScalarType.INTEGER]
                    ),
                    'my_table3': TypedDataFrame(
                        pd.DataFrame([[5], [6]], columns=['c']),
                        types=[BQScalarType.INTEGER]
                    ),
                    'my_table4': TypedDataFrame(
                        pd.DataFrame([[7], [8]], columns=['c']),
                        types=[BQScalarType.INTEGER]
                    ),
                }
            }
        })

    def test_ast_repr(self):
        # type: () -> None
        '''Check base AST Node's string representation'''
        node = AbstractSyntaxTreeNode()
        self.assertEqual(node.__repr__(), "AbstractSyntaxTreeNode()")

    def test_ast_strexpr(self):
        # type: () -> None
        '''Check base AST Node's prefix expression'''
        node = AbstractSyntaxTreeNode()
        self.assertEqual(node.strexpr(), '(ABSTRACTSYNTAXTREENODE )')

    def test_null_repr(self):
        # type: () -> None
        '''Check null's string representation'''
        node = EMPTY_NODE
        self.assertEqual(node.__repr__(), '_EmptyNode()')

    def test_null_strexpr(self):
        # type: () -> None
        '''Check null's prefix expression'''
        node = EMPTY_NODE
        self.assertEqual(node.strexpr(), 'null')

    def test_evaluation_context(self):
        # type: () -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)

        self.assertEqual(ec.table_ids, set(['my_table']))
        self.assertEqual(ec.column_to_table_ids, {'a': ['my_table']})
        self.assertEqual(ec.canonical_column_to_type, {'my_table.a': BQScalarType.INTEGER})
        self.assertEqual(ec.table.to_list_of_lists(), [[1], [2]])
        self.assertEqual(list(ec.table.dataframe), ['my_table.a'])
        self.assertEqual(ec.table.types, [BQScalarType.INTEGER])

    @data(
        (('my_table2', 'b'),),
        (('b',),)
    )
    @unpack
    def test_evaluation_context_path(self, path):
        # type: (Tuple[str, ...]) -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)

        self.assertEqual(ec.get_canonical_path(path), ('my_table2', 'b'))

    @data(
        (('c',), "field c is not present in any from'ed tables"),
        (('my_table', 'b'), "field b is not present in table my_table"),
        (('a',), r"field a is ambiguous: present in \[\('my_table', 'a'\), \('my_table2', 'a'\)\]")
    )
    @unpack
    def test_evaluation_context_path_error(self, path, error):
        # type: (Tuple[str, ...], str) -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)

        with self.assertRaisesRegexp(ValueError, error):
            ec.get_canonical_path(path)

    def test_context_lookup_success(self):
        # type: () -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)

        path = ('b', )
        result = ec.lookup(path)
        self.assertEqual(list(result.series), [2, 4])
        self.assertEqual(result.type_, BQScalarType.INTEGER)

    @data(
        (('b', ), [2, 4]),
        (('my_table2', 'b'), [2, 4]),
        (('c', ), [5, 6]),
    )
    @unpack
    def test_subcontext_lookup_success(self, path, expected_result):
        # type: (Tuple[str, ...], List[int]) -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)
        subcontext = EvaluationContext(self.table_context)
        subcontext.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table3')),
                                       EMPTY_NODE)
        ec.add_subcontext(subcontext)

        result = ec.lookup(path)
        self.assertEqual(list(result.series), expected_result)
        self.assertEqual(result.type_, BQScalarType.INTEGER)

    def test_subcontext_lookup_error_already_has_subcontext(self):
        # type: () -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)
        subcontext1 = EvaluationContext(self.table_context)
        subcontext1.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table3')),
                                        EMPTY_NODE)
        ec.add_subcontext(subcontext1)
        subcontext2 = EvaluationContext(self.table_context)
        subcontext2.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table4')),
                                        EMPTY_NODE)
        with self.assertRaisesRegexp(ValueError, 'Context already has subcontext'):
            ec.add_subcontext(subcontext2)

    def test_context_lookup_key_error(self):
        # type: () -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table')),
                               EMPTY_NODE)

        path = ('my_table', 'c')
        error = (r"path \('my_table', 'c'\) \(canonicalized to key 'my_table.c'\) "
                 r"not present in table; columns available: \['my_table.a'\]")

        # Fake in an entry to the column -> table mapping to test the error
        ec.column_to_table_ids['c'] = ['my_table']
        with self.assertRaisesRegexp(KeyError, error):
            ec.lookup(path)

    def test_context_lookup_type_error(self):
        # type: () -> None
        ec = EvaluationContext(self.table_context)
        ec.add_table_from_node(TableReference(('my_project', 'my_dataset', 'my_table2')),
                               EMPTY_NODE)

        path = ('my_table2', 'b')
        error = (r"path \('my_table2', 'b'\) \(canonicalized to key 'my_table2.b'\) "
                 r"not present in type dict; columns available: \['my_table2.a'\]")

        # Delete the type mapping for this column to test the error
        del ec.canonical_column_to_type['my_table2.b']
        with self.assertRaisesRegexp(KeyError, error):
            ec.lookup(path)


if __name__ == '__main__':
    unittest.main()
