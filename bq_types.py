# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""BigQuery types.

This module models the BigQuery type system with classes derived from BQType, and provides methods
to convert among BigQuery types, standard Python types, and NumPy types.

This module also provides two classes representing the basic data types used in the fake BQ backend:

    TypedSeries represents a column of data with the corresponding BigQuery type.
    TypedDataFrame represents a table of data, with each column's corresponding type.
"""

import datetime
import enum
from abc import ABCMeta, abstractmethod
from typing import (Any, AnyStr, Callable, Dict, List, Optional, Sequence, Tuple,  # noqa: F401
                    Type, Union, cast)

import numpy as np
import pandas as pd
import six
from google.cloud.bigquery.schema import SchemaField

# PythonType represents the set of Python types that correspond to BigQuery types.
# These types will be returned back out of the fake BQ to the calling Python code.
NoneType = type(None)
ScalarPythonType = Union[bool, datetime.date, datetime.datetime, int, float, str, NoneType,
                         six.text_type]
PythonType = Union[ScalarPythonType, Tuple]

# These types are possible values in a Pandas DataFrame or Series.
# Compare to PythonType above: numeric and boolean types use NumPy types, time types use
# Pandas Timestamp objects (not NumPy datetime64), and all other types stay as Python types.
PandasType = Union[np.bool_, pd.Timestamp, np.int64, np.float64, str, List[ScalarPythonType]]


class BQType(object):
    """Representation of a BigQuery column data type."""

    __metaclass__ = ABCMeta

    def to_dtype(self):
        # type: () -> np.dtype
        """Converts this BigQuery type to a NumPy dtype.

        Returns:
           'object' dtype, meaning NumPy will not try to interpret the type and will leave it
           as a Python object.

           Subclasses may override this function to return a more specific NumPy dtype.
        """
        return np.dtype('object')

    @classmethod
    def from_schema_field(cls, field):
        # type: (SchemaField) -> BQType
        """Converts from a BigQuery SchemaField object to a BQType subclass.

        This is a factory function, that constructs an object of the appropriate child class.

        Args:
            field: A BigQuery SchemaField object, the google cloud bigquery Python API
            representation of a column type.

        Returns:
            An instance of a BQType subclass that corresponds to the input type.
        """
        if field.mode in ('ARRAY', 'REPEATED'):
            return BQArray(BQScalarType.from_string(field.field_type))
        return BQScalarType.from_string(field.field_type)

    @abstractmethod
    def to_schema_field(self, name):
        # type: (str) -> SchemaField
        """Converts this type to a BigQuery SchemaField.

        Args:
            name: The name of the column.  This class represents a type; SchemaField represents
            a column, so it includes the type and also the name of the column.
        Returns:
            A SchemaField object corresponding to a column containing this class' type.

        This abstract method always raises NotImplementedError; child classes will override
        with an appropriate implementation.
        """

    @abstractmethod
    def convert(self, element):
        # type: (PandasType) -> PythonType
        """Converts a pandas Series element to a Python type corresponding to this BigQuery type.

        Args:
            element: One cell of a Pandas DataFrame or Series.  Will have numpy types like np.int64
            rather than a Python type like int.

        Returns:
            The element's value, cast to a corresponding Python type.
        """


class AbstractEnumMeta(enum.EnumMeta, ABCMeta):
    """A metaclass that combines Abstract Base Class with Enum.

    BQScalarType needs to explicitly create and use this shared metaclass, as Python requires that
    "the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its
    bases."  Otherwise, Python would have to choose whether to be an Abstract class or an Enum,
    as each class can only have a single metaclass, and it refuses to and throws a TypeError.
    """


# This is rather like gcp_base.BQFieldTypes, but I'd prefer it to be an actual
# Enum for better type-safety.
class BQScalarType(BQType, enum.Enum):
    """Representation of basic (scalar) BigQuery types as an enum."""

    __metaclass__ = AbstractEnumMeta

    BOOLEAN = 'BOOLEAN'
    DATE = 'DATE'
    DATETIME = 'DATETIME'
    INTEGER = 'INTEGER'
    FLOAT = 'FLOAT'
    STRING = 'STRING'
    TIMESTAMP = 'TIMESTAMP'

    def to_dtype(self):
        # type: () -> np.dtype
        """Converts this BigQuery type to a NumPy dtype.

        Returns:
           A NumPy dtype corresponding to this BigQuery type (e.g. np.dtype('int64') for INTEGER)
        """
        return np.dtype(_BQ_SCALAR_TYPE_TO_NUMPY_TYPE[self])

    @classmethod
    def from_string(cls, typename):
        # type: (str) -> BQScalarType
        """Reads this type from a string representation.

        A Factory method constructing an instance of BQScalarType corresponding to typename.
        The reason for not just using the inherited Enum constructor is to allow Standard BigQuery
        typenames to be aliases for Legacy typenames.

        Args:
            typename: A BigQuery type name, either Legacy (INTEGER, FLOAT, ...) or Standard
            (INT64, FLOAT64, ...).

        Returns:
            The corresponding BQScalarType enum.
        """
        if typename in _LEGACY_BQ_SCALAR_TYPE_FROM_BQ_SCALAR_TYPE:
            return _LEGACY_BQ_SCALAR_TYPE_FROM_BQ_SCALAR_TYPE[typename]
        return cls(typename)

    def to_schema_field(self, name):
        # type: (str) -> SchemaField
        """Converts this type to a BigQuery SchemaField.

        Args:
            name: The name of the column.  This class represents a type; SchemaField represents
            a column, so it includes the type and also the name of the column.

        Returns:
            A SchemaField object corresponding to a column containing this class' type.
        """
        return SchemaField(name=name, field_type=self.value)

    def __repr__(self):
        return 'BQScalarType.{}'.format(self.value)

    def convert(self, element):
        # type: (PandasType) -> ScalarPythonType
        """Converts a pandas Series element to a Python type corresponding to this BigQuery type.

        Args:
            element: One cell of a Pandas DataFrame or Series.  Will have numpy types like np.int64
            rather than a Python type like int.

        Returns:
            The element's value, cast to a corresponding Python type.
        """
        if pd.isnull(element):
            return None
        return _BQ_SCALAR_TYPE_TO_PYTHON_TYPE[self](element)


_LEGACY_BQ_SCALAR_TYPE_FROM_BQ_SCALAR_TYPE = {
    'BOOL': BQScalarType.BOOLEAN,
    'INT64': BQScalarType.INTEGER,
    'FLOAT64': BQScalarType.FLOAT,
}

_BQ_SCALAR_TYPE_TO_NUMPY_TYPE = {
    # integers are read as floats so that NULLs can turn into
    # np.nan (NaN)s without throwing an "Integer column has NA
    # values" ValueError
    BQScalarType.INTEGER: np.float64,
    BQScalarType.DATE: 'datetime64[ns]',
    BQScalarType.DATETIME: 'datetime64[ns]',
    BQScalarType.TIMESTAMP: 'datetime64[ns]',
    BQScalarType.STRING: np.string_,
    BQScalarType.FLOAT: np.float64,
    BQScalarType.BOOLEAN: np.bool_,
}


def _get_datetime(element):
    # type: (PandasType) -> datetime.datetime
    """Converts an element to a Python datetime.

    Args:
        element: A dataframe element; must be a Timestamp.

    Returns:
        The corresponding datatime.
    """
    assert isinstance(element, pd.Timestamp)
    return element.to_pydatetime()


def _get_date(timestamp):
    # type: (PandasType) -> datetime.date
    """Converts an element to a Python date.

    Args:
        element: A dataframe element; must be a Timestamp.

    Returns:
        The corresponding date.
    """
    assert isinstance(timestamp, pd.Timestamp)
    return timestamp.to_pydatetime().date()


def _get_str(s):
    # type: (PandasType) -> unicode
    """Converts an element to a Python string.

    Python 2 and Python 3 have different sets of string-related types, and different rules for
    conversion between those types.  The short version is, Python 2 has str and unicode, and strs
    are valid unicode values; Python 3 has str and bytes, and there is no implicit conversion
    between them.  An element that has the BQ type STRING might be any of these types, but it needs
    to end up being one of six.string_types, i.e. not bytes.  So: if it is a string type coming in,
    we leave it that way, if it's bytes, we do an explicit unicode conversion, and otherwise, it's
    an error.

    Args:
        element: A dataframe element; must be a string or bytes

    Returns:
        The corresponding string
    """
    if isinstance(s, six.string_types):
        return s
    if isinstance(s, bytes):
        return s.decode('utf-8')
    raise ValueError("Invalid string {}".format(s))


_BQ_SCALAR_TYPE_TO_PYTHON_TYPE = {
    BQScalarType.BOOLEAN: bool,
    BQScalarType.DATE: _get_date,
    BQScalarType.DATETIME: _get_datetime,  # timezone unaware
    # The casts of int and float are needed because mypy knows that they only actually work
    # when passed numbers.  Since we are tracking types via BQType, they should only be passed
    # numbers, and so the TypeError that would be raised if passed a non-number is a fine way
    # to handle what is an internal bug in fake BQ.
    BQScalarType.INTEGER: cast(Callable[[PandasType], int], int),
    BQScalarType.FLOAT: cast(Callable[[PandasType], float], float),
    BQScalarType.STRING: _get_str,
    BQScalarType.TIMESTAMP: _get_datetime,  # timezone aware
}  # type: Dict[BQScalarType, Callable[[PandasType], ScalarPythonType]]


class BQStructType(BQType):
    '''Representation of a BigQuery STRUCT type.

    Note: the corresponding STRUCT *value* is represented as a tuple
    '''

    # A cache of objects that represent STRUCT types, mapping the specification of a type to a
    # type object.  This is used to ensure there is only one object representing each STRUCT type.
    _STRUCT_TYPE_OBJECTS = {}  # type: Dict[Tuple[Tuple[Optional[str], ...], Tuple[Optional[BQType], ...]], BQStructType]  # noqa: E501

    def __new__(cls, fields, types):
        # type: (Sequence[Optional[str]], Sequence[Optional[BQType]]) -> BQStructType
        """Ensures that there is only one instance of BQStruct per component type.

        Args:
            fields: A list of optional string field names
            types: A list of optional types.

        Returns:
            A singleton Struct type object containing the provided type_
        """
        key = (tuple(fields), tuple(types))
        if key not in cls._STRUCT_TYPE_OBJECTS:
            struct = super(BQStructType, cls).__new__(cls)
            struct.__init__(fields, types)
            cls._STRUCT_TYPE_OBJECTS[key] = struct
        return cls._STRUCT_TYPE_OBJECTS[key]

    def __init__(self, fields, types):
        # type: (Sequence[Optional[str]], Sequence[Optional[BQType]]) -> None
        if len(fields) != len(types):
            raise ValueError("length of fields and types don't match: {}!={}"
                             .format(len(fields), len(types)))
        self.fields = fields
        self.types = types

    def __repr__(self):
        return 'STRUCT<{}>'.format(','.join('{} {}'.format(field, type_)
                                            for field, type_ in zip(self.fields, self.types)))

    def to_schema_field(self, name):
        # type: (str) -> SchemaField
        """Converts this type to a BigQuery SchemaField.

        Args:
            name: The name of the column.  This class represents a type; SchemaField represents
            a column, so it includes the type and also the name of the column.

        Returns:
            A SchemaField object corresponding to a column containing this class' type.
        """
        raise NotImplementedError("SchemaField for STRUCT not implemented")

    def convert(self, element):
        # type: (PandasType) -> PythonType
        return tuple(type_.convert(subelt) for type_, subelt in zip(self.types, element))


ArrayableType = Union[BQScalarType, BQStructType]


class BQArray(BQType):
    """Representation of a BigQuery ARRAY type.

    The corresponding array *value* is represented as a tuple
    """
    _ARRAY_TYPE_OBJECTS = {}  # type: Dict[ArrayableType, BQArray]

    def __new__(cls, type_):
        # type: (ArrayableType) -> BQArray
        """Ensures that there is only one instance of BQArray per component type.

        Args:
            type: A scalar type object.

        Returns:
            A singleton Array type object containing the provided type_
        """
        if type_ not in cls._ARRAY_TYPE_OBJECTS:
            array = super(BQArray, cls).__new__(cls)
            array.__init__(type_)
            cls._ARRAY_TYPE_OBJECTS[type_] = array
        return cls._ARRAY_TYPE_OBJECTS[type_]

    def __init__(self, type_):
        # type: (ArrayableType) -> None
        if isinstance(type_, BQArray):
            raise ValueError("Cannot create arrays of arrays!")
        self.type_ = type_

    def to_schema_field(self, name):
        # type: (str) -> SchemaField
        """Converts this type to a BigQuery SchemaField.

        Args:
            name: The name of the column.  This class represents a type; SchemaField represents
            a column, so it includes the type and also the name of the column.

        Returns:
            A SchemaField object corresponding to a column containing this class' type.
        """
        if isinstance(self.type_, BQScalarType):
            return SchemaField(name=name, field_type=self.type_.value, mode='REPEATED')
        raise NotImplementedError("SchemaField for ARRAY of {} not implemented"
                                  .format(self.type_))

    def __repr__(self):
        return 'BQArray({})'.format(self.type_)

    def convert(self, element):
        # type: (PandasType) -> PythonType
        """Converts a pandas Series element to a Python type corresponding to this BigQuery type.

        Args:
            element: One cell of a Pandas DataFrame or Series.  Will have numpy types like np.int64
            rather than a Python type like int.

        Returns:
            The element's value, cast to a corresponding Python type.
        """

        # First, check if the element is null.  If it is, isnull will return true.
        # If it isn't, element is a tuple, and so pd.isnull will return an np.ndarray, which
        # can't be tested for truthiness directly, but we know therefore isn't null.
        isnull = pd.isnull(element)
        if not isinstance(isnull, np.ndarray) and isnull:
            return None
        if not isinstance(element, tuple):
            raise ValueError("Array typed object {!r} isn't a tuple".format(element))
        return tuple(self.type_.convert(subelement) for subelement in element)


class TypedSeries(object):
    """A typed column of data."""
    def __init__(self, series, type_):
        # type: (Union[pd.Series, pd.SeriesGroupBy], BQType) -> None
        self._series = series
        self._type = type_

    @property
    def series(self):
        # type: () -> Union[pd.Series, pd.SeriesGroupBy]
        """Returns just the column of data."""
        return self._series

    @property
    def type_(self):
        # type: () -> BQType
        """Returns just the type of the data."""
        return self._type

    @property
    def dataframe(self):
        # type: () -> pd.DataFrame
        """Returns the column of data cast to a one-column table."""
        return pd.DataFrame(self.series)

    @property
    def types(self):
        # type: () -> List[BQType]
        """Returns the data type cast to a one-element list."""
        return [self.type_]

    def __repr__(self):
        return 'TypedSeries({!r}, {!r})'.format(self.series, self.type_)

    def to_list(self):
        # type: () -> List[PythonType]
        """Returns the column as a list of Python-typed objects."""
        return [self.type_.convert(element) for element in self.series]


class TypedDataFrame(object):
    def __init__(self, dataframe, types):
        # type: (Union[pd.DataFrame, pd.DataFrameGroupBy], List[BQType]) -> None
        if isinstance(dataframe, pd.DataFrame) and len(dataframe.columns) != len(types):
            raise ValueError(
                "Trying to create TypedDataFrame with mismatching type set; columns {!r} types {!r}"
                .format(list(dataframe.columns), types))
        self._dataframe = dataframe
        self._types = types

    @property
    def dataframe(self):
        # type: () -> Union[pd.DataFrame, pd.DataFrameGroupBy]
        """Returns the underlying DataFrame.

        This is a property so that it's immutable.

        Returns:
           The actual tabular data.
        """
        return self._dataframe

    @property
    def types(self):
        # type: () -> List[BQType]
        """Returns the data types, in the same order as the table's columns."""
        return self._types

    def __repr__(self):
        return 'TypedDataFrame({!r}, {!r})'.format(self.dataframe, self.types)

    def to_bq_schema(self):
        # type: () -> List[SchemaField]
        """Returns a BigQuery schema (list of schema fields) matching this object's types."""
        return [type_.to_schema_field(name)
                for name, type_ in zip(self.dataframe.columns, self.types)]

    def to_list_of_lists(self):
        # type: () -> List[List[PythonType]]
        """Returns the data as a list of rows, each row a list of Python-typed objects."""
        rows = []
        for unused_index, row in self.dataframe.iterrows():
            rows.append([type_.convert(element) for element, type_ in zip(list(row), self.types)])
        return rows


def _coerce_names(names):
    # type: (Sequence[str]) -> str
    '''Coerce a set of field names.  Names agree if equal or if one is None


    This function is called in the context of coercing STRUCT types like
    STRUCT<a INTEGER, b> with STRUCT(1 as a, 7).  In that case, it would be called twice, one for
    the first column (to merge the two 'a's into 'a') and once for the second column (to merge 'b'
    with the unnamed column in the second type).

    Args:
        names: A sequence of names (strings).  These are all names for the *same* field for
            different STRUCT types that are being coerced to a common type.
    Raises:
        ValueError if the names cannot be coerced (if two of the strings are both non-None and
        different).
    Returns:
        The single name matching all the names, or None if no non-empty names were provided.
    '''
    nonempty_names = {name for name in names if name is not None}
    if not nonempty_names:
        return None
    if len(nonempty_names) > 1:
        raise ValueError("Cannot merge Structs; field names {} do not match"
                         .format(nonempty_names))
    return nonempty_names.pop()


def _coerce_structs(struct_types):
    # type: (Sequence[BQStructType]) -> BQStructType
    '''Coerce a sequence of struct types into one.

    Struct types are merged field-by-field.  If the number of fields is different, they don't match.
    Two fields are merged by merging the names (see _coerce_names) and the types (recursively).

    Args:
        struct_types: a sequence of struct types.
    Raises:
        ValueError: if the types cannot be coerced.
    Returns:
        A single type that matches all the provided types.
    '''
    if not struct_types:
        return None

    num_fieldses = [len(type_.fields) for type_ in struct_types]
    if not all(num_fields == num_fieldses[0] for num_fields in num_fieldses[1:]):
        raise ValueError("Cannot merge types {}; number of fields varies!"
                         .format(struct_types))
    num_fields = num_fieldses[0]  # Now that we know they're all equal.

    field_types = [implicitly_coerce(*[type_.types[i] for type_ in struct_types])
                   for i in range(num_fields)]
    field_names = [_coerce_names([type_.fields[i] for type_ in struct_types])
                   for i in range(num_fields)]
    return BQStructType(field_names, field_types)


def implicitly_coerce(*types):
    # type: (BQType) -> BQType
    '''Given some number of types, return their common supertype, if any.
    All given types must be implicitly coercible to a common supertype.
    Specifically, INT64 and NUMERIC coerce to FLOAT, and STRING coerces to DATE or TIMESTAMP.
    All other conversions must be specified explicitly.

    See: https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_rules
    And: https://cloud.google.com/bigquery/docs/reference/standard-sql/conditional_expressions

    Note that there is no BQScalarType for NUMERIC - it is not supported in Fake BigQuery.

    Args:
        types: Types to combine
    Returns:
        A supertype to which all of the given types can be coerced
    '''
    # Filter out None types (the type of NULL)
    types = tuple(type_ for type_ in types if type_ is not None)
    if len(types) == 0:
        raise ValueError("No types provided to merge")
    if len(types) == 1:
        return types[0]
    if all(type_ == types[0] for type_ in types[1:]):
        return types[0]
    if all(type_ in [BQScalarType.INTEGER, BQScalarType.FLOAT] for type_ in types):
        return BQScalarType.FLOAT
    if all(type_ in [BQScalarType.STRING, BQScalarType.DATE] for type_ in types):
        return BQScalarType.DATE
    if all(type_ in [BQScalarType.STRING, BQScalarType.TIMESTAMP] for type_ in types):
        return BQScalarType.TIMESTAMP
    if all(isinstance(type_, BQStructType) for type_ in types):
        # cast is required due to mypy bug.
        return _coerce_structs(cast(Tuple[BQStructType, ...], types))
    raise ValueError("Cannot implicitly coerce the given types: {}".format(types))
