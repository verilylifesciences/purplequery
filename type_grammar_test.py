import unittest

from ddt import data, ddt, unpack

from bq_types import BQArray, BQScalarType, BQStructType
from tokenizer import tokenize
from type_grammar import array_type, scalar_type, struct_type


@ddt
class TypeGrammarTest(unittest.TestCase):

    @data(
        dict(text='INTEGER',
             expected_type=BQScalarType.INTEGER),
    )
    @unpack
    def test_scalar_type(self, text, expected_type):
        # type: (str, BQScalarType) -> None
        node, leftover = scalar_type(tokenize(text))
        self.assertFalse(leftover)
        self.assertEqual(node, expected_type)

    @data(
        dict(text='ARRAY<INTEGER>',
             expected_type=BQArray(BQScalarType.INTEGER)),
        dict(text='ARRAY<STRUCT<a INTEGER, b INTEGER> >',
             expected_type=BQArray(BQStructType(['a', 'b'],
                                                [BQScalarType.INTEGER, BQScalarType.INTEGER]))),
        dict(text='ARRAY<STRUCT<INTEGER, INTEGER> >',
             expected_type=BQArray(BQStructType([None, None],
                                                [BQScalarType.INTEGER, BQScalarType.INTEGER]))),
    )
    @unpack
    def test_array_type(self, text, expected_type):
        # type: (str, BQArray) -> None
        node, leftover = array_type(tokenize(text))
        self.assertFalse(leftover)
        self.assertEqual(node, expected_type)

    @data(
        dict(text='STRUCT<a INTEGER>',
             expected_type=BQStructType(['a'], [BQScalarType.INTEGER])),
        dict(text='STRUCT<INTEGER>',
             expected_type=BQStructType([None], [BQScalarType.INTEGER])),
        dict(text='STRUCT<a INTEGER, b STRING>',
             expected_type=BQStructType(['a', 'b'], [BQScalarType.INTEGER,
                                                     BQScalarType.STRING])),
        dict(text='STRUCT<INTEGER, b STRING>',
             expected_type=BQStructType([None, 'b'], [BQScalarType.INTEGER,
                                                      BQScalarType.STRING])),
        dict(text='STRUCT<ARRAY<FLOAT>, b STRING>',
             expected_type=BQStructType([None, 'b'], [BQArray(BQScalarType.FLOAT),
                                                      BQScalarType.STRING])),
    )
    @unpack
    def test_struct_type(self, text, expected_type):
        # type: (str, BQStructType) -> None
        node, leftover = struct_type(tokenize(text))
        self.assertFalse(leftover)
        self.assertEqual(node, expected_type)


if __name__ == '__main__':
    unittest.main()
