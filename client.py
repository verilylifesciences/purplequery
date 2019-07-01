# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Fake implementation of Google BigQuery client."""

import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, cast  # noqa: F401
from typing.io import TextIO  # noqa: F401

import numpy as np
import pandas as pd
import six
from google.api_core.exceptions import BadRequest, NotFound
from google.api_core.retry import Retry  # noqa: F401
from google.cloud.bigquery import Dataset, DatasetReference, Table, TableReference  # noqa: F401
from google.cloud.bigquery.dataset import DatasetListItem  # noqa: F401
from google.cloud.bigquery.job import LoadJobConfig, QueryJobConfig  # noqa: F401
from google.cloud.bigquery.schema import SchemaField  # noqa: F401
from google.cloud.bigquery.table import TableListItem  # noqa: F401

from bq_types import BQScalarType  # noqa: F401
from bq_types import BQArray, BQType, TypedDataFrame
from query import execute_query


class Client:
    """Fake implementation of Google BigQuery client.

    Method signatures in this class correspond to methods in
    google.cloud.bigquery.Client, with functionality implemented in-memory.  Not
    all parameters are used by this fake implementation.
    """

    def __init__(self, project):
        # type: (str) -> None
        """Constructs an instance of the fake client.

        Args:
            project: default project to use when no explicit project is specified.
        """

        self.project = project
        # This field holds all the data stored in the BigQuery fake.
        # It is a triply nested dictionary, mapping project id -> dataset id ->
        # table id -> a pair of a pandas DataFrame and the corresponding
        # BigQuery schema.
        self._datasets = {project: {}}  # type: Dict[str, Dict[str, Dict[str, TypedDataFrame]]]

    def _safe_lookup(self, project, dataset_id=None, table_id=None):
        # type: (str, Optional[str], Optional[str]) -> Any
        """Look up data in self._datasets, raise NotFound if the key(s) is/are not present.

        Can be used to look up a project, dataset, or table.

        Args:
            project: project to look up
            dataset_id: If not specified, return datasets in project.
                        If specified, the dataset to look up.
            table_id: If not specified, return tables in datasets.
                      If specified, the table to look up.
        Returns:
            The map of all datasets in a requested project OR
            the map of all tables in a requested dataset OR
            a specific table
            depending on the specificity of the arguments.
        """
        project_map = self._datasets

        # First look up the project
        if project not in project_map:
            raise NotFound("Project {} not found".format(project))
        dataset_map = project_map[project]

        # If requested, look up the dataset
        if dataset_id is None:
            return dataset_map
        if dataset_id not in dataset_map:
            raise NotFound("Dataset {} not found".format(dataset_id))
        table_map = dataset_map[dataset_id]

        # If requested, look up a table.
        if table_id is None:
            return table_map
        if table_id not in table_map:
            raise NotFound("Table {} not found".format(table_id))
        return table_map[table_id]

    def dataset(self, dataset_id, project=None):
        # type: (str, Optional[str]) -> DatasetReference
        """Constructs a reference to a dataset.

        Args:
            dataset_id: Dataset to look up
            project: If specified, project to find the dataset in, otherwise client's default.

        Returns:
            Reference to the dataset found.
        """
        project = project or self.project
        return DatasetReference(project, dataset_id)

    def get_dataset(self, dataset_ref, retry=None):
        # type: (DatasetReference, Optional[Retry]) -> Dataset
        """Looks up a dataset.

        Args:
            dataset_ref: Dataset reference to find.
            retry: If provided, what retry strategy to use (unused in this implementation).

        Returns:
            Dataset found.
        """
        del retry  # Unused in this implementation.
        # Make sure project and dataset exist
        self._safe_lookup(dataset_ref.project, dataset_ref.dataset_id)
        return Dataset(dataset_ref)

    def list_datasets(self, project=None, max_results=None, retry=None):
        # type: (Optional[str], Optional[int], Optional[Retry]) -> List[DatasetListItem]
        """Lists all datasets in a given project, or in the default project.

        Args:
            project: Project to list, otherwise uses the client's default
            max_results: If provided, how many results to use (unused in this implementation).
            retry: If provided, what retry strategy to use (unused in this implementation).

        Returns:
            References to all datasets found.  Full Dataset objects are not
            provided (in the real API) for performance reasons.
        """
        del retry  # Unused in this implementation.
        del max_results  # Unused in this implementation.
        project = project or self.project
        # Construct DatasetListItems by passing an abbreviated JSON object as
        # found in the "datasets" member of the return value of
        # https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/list
        return [
            DatasetListItem({"datasetReference": {"projectId": project, "datasetId": dataset_id}})
            for dataset_id in self._safe_lookup(project)]

    def list_tables(self, dataset_ref, retry=None):
        # type: (DatasetReference, Optional[Retry]) -> List[TableListItem]
        """Lists all tables in a given dataset.

        Args:
            dataset_ref: The dataset to list.
            retry: If provided, what retry strategy to use (unused in this implementation).

        Returns:
            References to all tables found.  Full Table objects are not provided (in the real API)
            for performance reasons.
        """
        del retry  # Unused in this implementation.
        # Construct TableListItems by passing an abbreviated JSON object as
        # found in the "tables" member of the return value of
        # https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/list
        return [
            TableListItem({"tableReference":
                           {"projectId": dataset_ref.project,
                            "datasetId": dataset_ref.dataset_id,
                            "tableId": table_id}})
            for table_id in self._safe_lookup(dataset_ref.project, dataset_ref.dataset_id)]

    def create_dataset(self, dataset):
        # type: (Dataset) -> None
        """Creates a dataset.

        Args:
            dataset: Dataset to create.
        """
        if dataset.project not in self._datasets:
            self._datasets[dataset.project] = {}
        if dataset.dataset_id in self._datasets[dataset.project]:
            raise ValueError("Dataset {} already exists".format(dataset))
        self._datasets[dataset.project][dataset.dataset_id] = {}

    def get_table(self, table_ref, retry=None):
        # type: (TableReference, Optional[Retry]) -> Table
        """Looks up a table.

        Args:
            table_ref: Table reference to find.
            retry: If provided, what retry strategy to use (unused in this implementation).

        Returns:
            Table found.
        """
        del retry  # Unused in this implementation.
        typed_dataframe = self._safe_lookup(
                table_ref.project, table_ref.dataset_id, table_ref.table_id)
        return Table(table_ref, typed_dataframe.to_bq_schema())

    def create_table(self, table):
        # type: (Table) -> None
        """Creates a table.

        Args:
            table: Table to create.
        """
        table_map = self._safe_lookup(table.project, table.dataset_id)
        if table.table_id in table_map:
            raise ValueError("Table {} already exists".format(table))
        bq_types = [BQType.from_schema_field(field) for field in table.schema]
        table_map[table.table_id] = TypedDataFrame(
            pd.DataFrame(
                data=collections.OrderedDict([(field.name, pd.Series([], dtype=bq_type.to_dtype()))
                                              for field, bq_type in zip(table.schema, bq_types)])),
            bq_types)

    def get_table_dataframe(self, project, dataset_id, table_id):
        # type: (str, str, str) -> pd.DataFrame
        """Used in fake_gcp for reading a table as a dataframe.
        This method is not in the Google BigQuery Client API.

        Args:
            project: Project ID of requested table
            dataset_id: Dataset ID of requested table
            table_id: Table ID

        Returns:
            The DataFrame representation of this table.
        """
        dataframe, schema = self._safe_lookup(project, dataset_id, table_id)
        return dataframe

    def set_table_dataframe(self, dataframe, project, dataset_id, table_id):
        # type: (pd.DataFrame, str, str, str) -> None
        """Used in fake_gcp for modifying a table from a dataframe.
        Assumes the table already exists and is only being updated!
        This method is not in the Google BigQuery Client API.

        Args:
            dataframe: Pandas DataFrame to set as table
            project: Project ID of table to set
            dataset_id: Dataset ID of table to set
            table_id: Table ID - assumed to exist
        """
        old_typed_dataframe = self._safe_lookup(project, dataset_id, table_id)
        self._datasets[project][dataset_id][table_id] = TypedDataFrame(dataframe,
                                                                       old_typed_dataframe.schema)

    def load_table_from_file(self, fileobj, table_ref, job_config, rewind):
        # type: (TextIO, TableReference, LoadJobConfig, bool) -> _FakeJob
        """Loads a table from a file.

        Args:
            fileobj: A file-like object containing the data.
            table_ref: The (already existing) table into which to load the data.
            job_config: A configuration object giving job info, such as file format.
                Ignored in this implementation.
            rewind: Whether to seek to the beginning of the file before reading it.
        Returns:
            A Job object that can be waited on for completion of the load.
        """
        del job_config  # Ignored in this implementation.
        typed_dataframe = self._safe_lookup(table_ref.project,
                                            table_ref.dataset_id,
                                            table_ref.table_id)
        if rewind:
            fileobj.seek(0)

        # Before we can call pandas.read_csv, we need to set up special handling for two things:
        # dates, and array(list) fields.

        # First, we find all the fields that are of a datetime type, tell read_csv that they're
        # of a string dtype instead, but then add their column names to the list of date_fields
        # that read_csv knows how to parse separately.  Why read_csv can't do this itself,
        # I have no idea.
        dtypes = typed_dataframe.dataframe.dtypes.to_dict()
        date_fields = []
        for column, dtype in six.iteritems(dtypes):
            if dtype == 'datetime64[ns]':
                dtypes[column] = str
                date_fields.append(column)

        # Next, we find any fields of Array type.  For this, we declare that an array-valued
        # field is serialized into a csv as a (quoted) comma-separated list, and so we set up
        # a type-specific "converter" function to read that back into the appropriate list type.
        converters = {}
        columns = list(typed_dataframe.dataframe.columns)
        for column, bq_type in zip(columns, typed_dataframe.types):
            if isinstance(bq_type, BQArray):
                converters[column] = _make_array_reader(bq_type.type_)

        # Having set up our special type handling, we now call read_csv and load it into the
        # desired table.
        self._safe_lookup(table_ref.project, table_ref.dataset_id)[table_ref.table_id] = (
            TypedDataFrame(
                pd.read_csv(
                    fileobj,
                    header=None,
                    names=columns,
                    dtype=dtypes,
                    parse_dates=date_fields,
                    converters=converters),
                typed_dataframe.types))
        return _FakeJob(None)

    def delete_dataset(self, dataset_ref, retry=None, delete_contents=False):
        # type: (DatasetReference, Optional[Retry], bool) -> None
        """Deletes a dataset.

        Args:
            dataset_ref: The dataset reference to delete.
            retry: If provided, what retry strategy to use (unused in this implementation).
            delete_contents: If true, delete the contents of the dataset first.  If false and the
                dataset is nonempty, raise an error.
        """
        del retry  # Unused in this implementation.
        if self._safe_lookup(dataset_ref.project, dataset_ref.dataset_id) and not delete_contents:
            raise BadRequest("Can't delete dataset {}; dataset is not empty".format(dataset_ref))
        del self._datasets[dataset_ref.project][dataset_ref.dataset_id]

    def delete_table(self, table_ref, retry=None):
        # type: (TableReference, Optional[Retry]) -> None
        """Deletes a table.

        Args:
            table_ref: The table reference to delete.
            retry: If provided, what retry strategy to use (unused in this implementation).
        """
        del retry  # Unused in this implementation.
        # Make sure table exists before deleting
        self._safe_lookup(table_ref.project, table_ref.dataset_id, table_ref.table_id)
        del self._datasets[table_ref.project][table_ref.dataset_id][table_ref.table_id]

    def insert_rows(self, table, rows, retry=None):
        # type: (Table, List[Dict[str, Any]], Optional[Retry]) -> List[str]
        """Appends additional rows to an existing table.

        Args:
            table: The Table to modify.
            rows: A list of dictionaries mapping column name to values.
            retry: If provided, what retry strategy to use (unused in this implementation).

        Returns:
            A list of errors encountered, if any.
        """
        del retry  # Unused in this implementation.
        new_dataframe_rows = []
        for row in rows:
            new_dataframe_rows.append([row[field.name] for field in table.schema])
        new_dataframe = pd.DataFrame(new_dataframe_rows,
                                     columns=[field.name for field in table.schema])
        old_typed_dataframe = self._safe_lookup(table.project, table.dataset_id, table.table_id)
        self._datasets[table.project][table.dataset_id][table.table_id] = TypedDataFrame(
            _rename_and_append_dataframe(
                old_typed_dataframe.dataframe, new_dataframe),
            old_typed_dataframe.types)
        return []  # no errors

    def query(self, query, job_config, retry=None):
        # type: (str, QueryJobConfig, Optional[Retry]) -> _FakeJob
        """Executes an SQL query.

        Args:
            query: A BigQuery SQL string.
            job_config: A configuration object.  See real API for documentation; important
                behavior includes whether and how to write the results of the query to a new
                or existing table.
            retry: If provided, what retry strategy to use (unused in this implementation).
        Returns:
            A Job object that can be waited on.  When complete, the result is a
            List of Row objects, containing a list of Python datatypes corresponding to the query.
        """
        del retry  # Unused in this implementation.
        if job_config.use_legacy_sql:
            raise NotImplementedError("Legacy SQL syntax is not implemented.")

        result = execute_query(query, self._datasets)
        if job_config.destination:
            table_ref = job_config.destination
            table_map = self._safe_lookup(table_ref.project, table_ref.dataset_id)
            typed_dataframe = table_map.get(table_ref.table_id, TypedDataFrame(pd.DataFrame(), []))
            if job_config.write_disposition not in ('WRITE_EMPTY', 'WRITE_APPEND',
                                                    'WRITE_TRUNCATE'):
                raise ValueError("Unsupported write write_disposition {}".format(
                        job_config.write_disposition))

            if (job_config.write_disposition == 'WRITE_EMPTY' and
                    not typed_dataframe.dataframe.empty):
                raise ValueError(
                    "Bad request; trying to overwrite nonempty table {} with disposition {}"
                    .format(job_config.destination, job_config.write_disposition))
            elif (job_config.write_disposition == 'WRITE_APPEND' and
                  not typed_dataframe.dataframe.empty):
                # TODO: Support appending rows to a table that have a subset of the existing columns
                # by filling missing column values with NULLs.
                if typed_dataframe.types != result.types:
                    raise ValueError(
                        "Bad request; trying to append data of type {} to data of type {}".format(
                            result.types, typed_dataframe.types))
                table_map[table_ref.table_id] = (
                    TypedDataFrame(
                        _rename_and_append_dataframe(typed_dataframe.dataframe, result.dataframe),
                        typed_dataframe.types))
            else:  # Either write_truncate, or (write_empty or write_append) to an empty table
                table_map[table_ref.table_id] = result
        return _FakeJob([_FakeRow(row) for row in result.to_list_of_lists()])


def _rename_and_append_dataframe(old_dataframe, new_dataframe):
    # type: (pd.DataFrame, pd.DataFrame) -> pd.DataFrame
    """Append the rows of new_dataframe to old_dataframe.

    DataFrame.append requires not just that the number of columns match, but that the names of
    the columns match as well.  That won't necessarily be the case in the current implementation
    due to the way we rename columns to include the table name (e.g. table1.a, table1.b), so
    we explicitly rename the second dataframe's columns to match the first, and then append.

    Args:
       old_dataframe: A pandas DataFrame.
       new_dataframe: A pandas DataFrame, which we promise has equivalent columns in the same
           order, which will be appended to the first dataframe.
    Returns:
       A new dataframe having all of the first dataframe's rows, then all of the second's, and
       using the first dataframe's column names.
    """
    new_dataframe = new_dataframe.rename(
        columns=dict(zip(new_dataframe.columns, old_dataframe.columns)))
    return old_dataframe.append(new_dataframe)


def _make_array_reader(bq_type):
    # type: (BQScalarType) -> Callable[[str], Any]
    """Returns a converter function so pandas.read_csv can read array-valued cells."""
    numpy_type = bq_type.to_dtype().type

    def read_array_from_csv(s):
        # type: (str) -> Any
        """Reads a single csv cell and returns a list of values or NaN."""
        if s == 'NULL':
            return np.nan
        return [numpy_type(element) for element in s.split(',')]

    return read_array_from_csv


class _FakeJob:
    """A minimal fake implementation of google.cloud.bigquery.*Job."""
    def __init__(self, result):
        # type: (Any) -> None
        self._result = result
        self.error_result = None
        self.errors = ()

    def result(self, timeout=None):
        # type: (Optional[int]) -> Any
        """Returns the result of a complete job."""
        return self._result

    def done(self, retry=None):
        # type: (Optional[bool]) -> bool
        """Returns True when the Job is complete.  Fake jobs are instantly complete."""
        return True


class _FakeRow:
    """A minimal fake implementation of google.cloud.bigquery.Row."""
    def __init__(self, values):
        # type: (List[Any]) -> None
        self._values = tuple(values)

    def values(self):
        # type: () -> Tuple[Any, ...]
        """Return the values in the row."""
        return self._values
