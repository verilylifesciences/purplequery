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

import datetime
import unittest
from typing import List, Union  # noqa: F401

import numpy as np
import pandas as pd
from ddt import data, ddt, unpack
from google.cloud.bigquery.schema import SchemaField

from bq_types import PythonType  # noqa: F401
from bq_types import BQArray, BQScalarType, BQType, TypedDataFrame, TypedSeries, implicitly_coerce

# The NumPy types that are used to read in data into Pandas.
NumPyType = Union[np.bool_, np.datetime64, np.float64, np.string_]


@ddt
class BqTypesTest(unittest.TestCase):

    @data(
        (BQScalarType.BOOLEAN, SchemaField(name='foo', field_type='BOOLEAN')),
        (BQScalarType.DATE, SchemaField(name='foo', field_type='DATE')),
        (BQScalarType.DATETIME, SchemaField(name='foo', field_type='DATETIME')),
        (BQScalarType.INTEGER, SchemaField(name='foo', field_type='INTEGER')),
        (BQScalarType.FLOAT, SchemaField(name='foo', field_type='FLOAT')),
        (BQScalarType.STRING, SchemaField(name='foo', field_type='STRING')),
        (BQScalarType.TIMESTAMP, SchemaField(name='foo', field_type='TIMESTAMP')),
    )
    @unpack
    def test_convert_between_schema_field_and_bq_type(self, bq_type, schema_field):
        # type: (BQScalarType, SchemaField) -> None

        # Test scalar
        self.assertEqual(BQType.from_schema_field(schema_field), bq_type)
        self.assertEqual(bq_type.to_schema_field('foo'), schema_field)

        # Test array
        schema_array_field = SchemaField(
                name=schema_field.name,
                field_type=schema_field.field_type,
                mode='ARRAY')
        bq_array_type = BQArray(bq_type)
        self.assertEqual(BQType.from_schema_field(schema_array_field), bq_array_type)
        self.assertEqual(bq_array_type.to_schema_field('foo'), schema_array_field)

    @data(
        (BQScalarType.BOOLEAN, SchemaField(name='foo', field_type='BOOL')),
        (BQScalarType.INTEGER, SchemaField(name='foo', field_type='INTEGER')),
        (BQScalarType.FLOAT, SchemaField(name='foo', field_type='FLOAT')),
    )
    @unpack
    def test_convert_from_legacy_schema_field_to_bq_type(self, bq_type, schema_field):
        # type: (BQScalarType, SchemaField) -> None

        self.assertEqual(BQType.from_schema_field(schema_field), bq_type)

    @data(
        (BQScalarType.BOOLEAN,),
        (BQScalarType.DATE,),
        (BQScalarType.DATETIME,),
        (BQScalarType.INTEGER,),
        (BQScalarType.FLOAT,),
        (BQScalarType.STRING,),
        (BQScalarType.TIMESTAMP,),
    )
    @unpack
    def test_two_arrays_of_same_type_are_same_object(self, bq_type):
        # type: (BQScalarType) -> None
        # Type objects are immutable, and we need to be able to compare them
        # (an array of ints is an array of ints, but it's not a string or an array of floats).
        # A way to achieve this is to ensure that all types, including arrays, are singletons.
        # So we test that for each scalar type, creating two arrays of it yields the same object.
        a1 = BQArray(bq_type)
        a2 = BQArray(bq_type)
        self.assertIs(a1, a2)

    @data(
        (BQScalarType.BOOLEAN, np.bool_(True), True),
        (BQScalarType.DATE, np.datetime64('2019-01-07'), datetime.date(2019, 1, 7)),
        (BQScalarType.DATETIME, np.datetime64('2019-01-07T10:32:05.123456'),
         datetime.datetime(2019, 1, 7, 10, 32, 5, 123456)),
        (BQScalarType.INTEGER, np.float64(35.0), 35),
        (BQScalarType.FLOAT, np.float64(12.34), 12.34),
        (BQScalarType.STRING, np.string_('hello'), 'hello'),
        (BQScalarType.TIMESTAMP, np.datetime64('2019-01-07T10:32:05.123456'),
         datetime.datetime(2019, 1, 7, 10, 32, 5, 123456))
    )
    @unpack
    def test_convert(self, bq_type, np_object, py_object):
        # type: (BQScalarType, NumPyType, PythonType) -> None

        # First, convert from a NumPy-typed object to a Pandas-typed object.
        # Types are mostly the same except for np.datetime64 becomes pd.Timestamp
        # We do this by creating a Pandas Series containing the single object, and then
        # converting it to a sequence and extracting its single element.
        pd_object, = pd.Series(np_object)
        self.assertEqual(bq_type.convert(pd_object), py_object)

        # Test that for any type, a NaN converts to None
        self.assertIsNone(bq_type.convert(np.nan))

        # Now test the same conversion for a list (array) of objects.
        # Construct a Series containing a single row which is a list of three objects.
        pd_array_object, = pd.Series([[pd_object]*3])
        self.assertEqual(BQArray(bq_type).convert(pd_array_object), [py_object]*3)

        # Test that for any Array type, a NaN converts to None
        self.assertIsNone(BQArray(bq_type).convert(np.nan))

    @data(
        (BQScalarType.BOOLEAN, np.bool_),
        (BQScalarType.DATE, 'datetime64[ns]'),
        (BQScalarType.DATETIME, 'datetime64[ns]'),
        (BQScalarType.INTEGER, np.float64),
        (BQScalarType.FLOAT, np.float64),
        (BQScalarType.STRING, np.string_),
        (BQScalarType.TIMESTAMP, 'datetime64[ns]'),
    )
    @unpack
    def test_to_dtype(self, bq_type, np_type):
        # type: (BQScalarType, NumPyType) -> None
        self.assertEqual(bq_type.to_dtype(), np.dtype(np_type))
        # NumPy doesn't know from cell elements that are lists, so it just leaves it as an
        # uninterpreted Python object.
        self.assertEqual(BQArray(bq_type).to_dtype(), np.dtype('object'))

    def test_get_typed_series_as_list(self):
        typed_series = TypedSeries(
                pd.Series([[np.float64(1.5), np.float64(2.5), np.float64(3.0)],
                           [np.float64(2.5), np.float64(3.5), np.float64(4.0)]]),
                BQArray(BQScalarType.FLOAT))
        self.assertEqual(typed_series.to_list(),
                         [[1.5, 2.5, 3.0],
                          [2.5, 3.5, 4.0]])

    def test_get_typed_dataframe_schema(self):
        typed_dataframe = TypedDataFrame(pd.DataFrame(columns=['a', 'b']),
                                         [BQScalarType.BOOLEAN,
                                          BQArray(BQScalarType.FLOAT)])
        self.assertEqual(typed_dataframe.to_bq_schema(),
                         [SchemaField(name='a', field_type='BOOLEAN'),
                          SchemaField(name='b', field_type='FLOAT', mode='ARRAY')])

    def test_get_typed_dataframe_as_list_of_lists(self):
        typed_dataframe = TypedDataFrame(
                pd.DataFrame(
                        [[np.bool_(True), [np.float64(1.5), np.float64(2.5), np.float64(3.0)]],
                         [np.bool_(False), [np.float64(2.5), np.float64(3.5), np.float64(4.0)]]],
                        columns=['a', 'b']),
                [BQScalarType.BOOLEAN,
                 BQArray(BQScalarType.FLOAT)])
        self.assertEqual(typed_dataframe.to_list_of_lists(),
                         [[True, [1.5, 2.5, 3.0]],
                          [False, [2.5, 3.5, 4.0]]])

    @data(
        ([BQScalarType.INTEGER], BQScalarType.INTEGER),
        ([None, BQScalarType.INTEGER], BQScalarType.INTEGER),
        ([BQScalarType.FLOAT], BQScalarType.FLOAT),
        ([BQScalarType.FLOAT, None], BQScalarType.FLOAT),
        ([BQScalarType.FLOAT, BQScalarType.FLOAT], BQScalarType.FLOAT),
        ([BQScalarType.STRING, BQScalarType.STRING], BQScalarType.STRING),
        ([BQScalarType.STRING, None, BQScalarType.STRING], BQScalarType.STRING),
        ([BQScalarType.INTEGER, BQScalarType.FLOAT], BQScalarType.FLOAT),
        ([BQScalarType.STRING, BQScalarType.DATE], BQScalarType.DATE),
        ([BQScalarType.STRING, BQScalarType.TIMESTAMP], BQScalarType.TIMESTAMP),
    )
    @unpack
    def test_implicitly_coerce(self, input_types, expected_supertype):
        # type: (List[BQScalarType], BQScalarType) -> None
        supertype = implicitly_coerce(*input_types)
        self.assertEqual(supertype, expected_supertype)

    @data(
        ([], "No types provided to merge"),
        ([BQScalarType.STRING, BQScalarType.INTEGER],
         "Cannot implicitly coerce the given types:"),
        ([BQScalarType.STRING, BQScalarType.DATE, BQScalarType.TIMESTAMP],
         "Cannot implicitly coerce the given types:"),
    )
    @unpack
    def test_implicitly_coerce_error(self, input_types, error):
        # type: (List[BQScalarType], str) -> None
        with self.assertRaisesRegexp(ValueError, error):
            implicitly_coerce(*input_types)


if __name__ == '__main__':
    unittest.main()
