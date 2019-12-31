# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import csv
import datetime
import unittest
from typing import Any, List, Tuple  # noqa: F401

import six
from ddt import data, ddt, unpack
from google.api_core.exceptions import BadRequest, NotFound
from google.cloud.bigquery import Dataset, DatasetReference, Table, TableReference
from google.cloud.bigquery.job import QueryJobConfig
from google.cloud.bigquery.schema import SchemaField

from purplequery.bq_types import PythonType  # noqa: F401
from purplequery.client import _FakeJob  # noqa: F401
from purplequery.client import Client
from six.moves import cStringIO

_TEST_SCHEMA = [SchemaField(name="num", field_type='INTEGER'),
                SchemaField(name="ID", field_type='STRING'),
                SchemaField(name="height", field_type='FLOAT'),
                SchemaField(name="likes_chocolate", field_type='BOOLEAN'),
                SchemaField(name="start_date", field_type='DATE'),
                SchemaField(name="mid_date", field_type='DATETIME'),
                SchemaField(name="end_time", field_type='TIMESTAMP'),
                SchemaField(name="xs", field_type='INTEGER', mode='REPEATED'),
                ]


class ClientTestBase(unittest.TestCase):

    def setUp(self):
        self.bq_client = Client('my_project')

    def assertRowsExpected(self, query_job, expected_rows):
        # type: (_FakeJob, List[List[PythonType]]) -> None
        """Assert that query_job has finished and it contains the expected rows.

        Args:
            query_job: A QueryJob returned from query
            expected_rows: A List of lists of values
        """
        self.assertTrue(query_job.done())
        self.assertEqual(query_job.statement_type, 'SELECT')
        self.assertEqual(query_job.project, 'my_project')
        self.assertEqual(query_job.location, 'YourDesktop')
        self.assertTrue(query_job.job_id)
        rows = [list(row.values()) for row in query_job.result()]
        self.assertEqual(rows, expected_rows)


@ddt
class ClientTest(ClientTestBase):

    def assertDatasetReferenceEqual(self, expected_dataset_reference, found_dataset_reference):
        # type: (DatasetReference, DatasetReference) -> None
        self.assertEqual(expected_dataset_reference.project, found_dataset_reference.project)
        self.assertEqual(expected_dataset_reference.dataset_id, found_dataset_reference.dataset_id)

    def test_dataset_lookup(self):
        # type: () -> None
        with self.assertRaisesRegexp(NotFound, 'some_other_project'):
            self.bq_client.get_dataset(DatasetReference('some_other_project', 'dataset1'))
        with self.assertRaisesRegexp(NotFound, 'dataset1'):
            self.bq_client.get_dataset(DatasetReference('my_project', 'dataset1'))
        dataset1 = Dataset(DatasetReference('my_project', 'dataset1'))
        self.bq_client.create_dataset(dataset1)

        self.assertDatasetReferenceEqual(
            self.bq_client.get_dataset(DatasetReference('my_project', 'dataset1')).reference,
            dataset1.reference)
        self.assertDatasetReferenceEqual(self.bq_client.dataset('dataset1'),
                                         dataset1.reference)
        self.assertDatasetReferenceEqual(self.bq_client.dataset('dataset1', 'my_project'),
                                         dataset1.reference)

    def test_dataset_delete(self):
        # type: () -> None
        dataset1 = Dataset(DatasetReference('my_project', 'dataset1'))

        # Can't delete dataset, doesn't exist yet.
        with self.assertRaisesRegexp(NotFound, 'dataset1'):
            self.bq_client.delete_dataset(dataset1.reference)

        self.bq_client.create_dataset(dataset1)
        self.bq_client.create_table(Table(TableReference(dataset1.reference, 'table1'),
                                          _TEST_SCHEMA))

        # Can't delete dataset, not empty
        with self.assertRaises(BadRequest):
            self.bq_client.delete_dataset(dataset1.reference)

        # Okay to delete, specifically requesting to delete contents.
        self.bq_client.delete_dataset(dataset1.reference, delete_contents=True)

        # And now dataset is gone again.
        with self.assertRaisesRegexp(NotFound, 'dataset1'):
            self.bq_client.get_dataset(DatasetReference('my_project', 'dataset1'))

    def test_listing_datasets(self):
        # type: () -> None
        self.assertFalse(self.bq_client.list_datasets())
        self.assertFalse(self.bq_client.list_datasets('my_project'))
        self.bq_client.create_dataset(Dataset(DatasetReference('my_project', 'dataset1')))
        self.bq_client.create_dataset(Dataset(DatasetReference('my_project', 'dataset2')))
        self.bq_client.create_dataset(Dataset(DatasetReference('other_project', 'dataset3')))
        six.assertCountEqual(self,
                             [dataset.dataset_id for dataset in self.bq_client.list_datasets()],
                             ['dataset1', 'dataset2'])
        six.assertCountEqual(
                self,
                [dataset.dataset_id for dataset in self.bq_client.list_datasets('my_project')],
                ['dataset1', 'dataset2'])
        six.assertCountEqual(
                self,
                [dataset.dataset_id for dataset in self.bq_client.list_datasets('other_project')],
                ['dataset3'])

    def test_table_lookup(self):
        # type: () -> None
        dataset_ref1 = DatasetReference('my_project', 'dataset1')
        table_ref1 = TableReference(dataset_ref1, 'table1')
        table1 = Table(table_ref1, _TEST_SCHEMA)

        # Trying to get the same dataset/table in another project doesn't work.
        with self.assertRaisesRegexp(NotFound, 'other_project'):
            self.bq_client.get_table(
                TableReference(DatasetReference('other_project', 'dataset1'), 'table1'))

        # Trying to get the table before the dataset exists doesn't work
        with self.assertRaisesRegexp(NotFound, 'dataset1'):
            self.bq_client.get_table(table_ref1)
        self.bq_client.create_dataset(Dataset(dataset_ref1))

        # Trying to get the table before the table exists doesn't work
        with self.assertRaises(NotFound):
            self.bq_client.get_table(table_ref1)

        self.bq_client.create_table(table1)

        # Assert the created table has the expected properties.
        table_found = self.bq_client.get_table(table_ref1)
        self.assertEqual(table1.project, "my_project")
        self.assertEqual(table1.dataset_id, "dataset1")
        self.assertEqual(table1.table_id, "table1")
        six.assertCountEqual(self, table_found.schema, _TEST_SCHEMA)

    def test_delete_table(self):
        # type: () -> None
        dataset_ref1 = DatasetReference('my_project', 'dataset1')
        table_ref1 = TableReference(dataset_ref1, 'table1')

        # Can't delete table, dataset not created yet.
        with self.assertRaisesRegexp(NotFound, 'dataset1'):
            self.bq_client.delete_table(table_ref1)

        self.bq_client.create_dataset(Dataset(dataset_ref1))

        # Can't delete table, table not created yet.
        with self.assertRaisesRegexp(NotFound, 'table1'):
            self.bq_client.delete_table(table_ref1)

        table1 = Table(table_ref1, _TEST_SCHEMA)
        self.bq_client.create_table(table1)

        self.bq_client.delete_table(table_ref1)
        with self.assertRaisesRegexp(NotFound, 'table1'):
            self.bq_client.get_table(table_ref1)

    def test_listing_tables(self):
        # type: () -> None
        dataset_ref1 = DatasetReference('my_project', 'dataset1')
        self.bq_client.create_dataset(Dataset(dataset_ref1))
        self.bq_client.create_table(Table(TableReference(dataset_ref1, 'table1'), _TEST_SCHEMA))
        self.bq_client.create_table(Table(TableReference(dataset_ref1, 'table2'), []))
        six.assertCountEqual(
            self,
            [table_ref.table_id for table_ref in self.bq_client.list_tables(dataset_ref1)],
            ['table1', 'table2'])

    def test_listing_tables_with_max(self):
        # type: () -> None
        dataset_ref1 = DatasetReference('my_project', 'dataset1')
        self.bq_client.create_dataset(Dataset(dataset_ref1))
        for i in range(10):
            self.bq_client.create_table(Table(TableReference(dataset_ref1, 'table{}'.format(i)),
                                              _TEST_SCHEMA))
        self.assertEqual(5, len(self.bq_client.list_tables(dataset_ref1, max_results=5)))
        self.assertEqual(10, len(self.bq_client.list_tables(dataset_ref1, max_results=20)))
        self.assertEqual(10, len(self.bq_client.list_tables(dataset_ref1)))

    # This row uses _TEST_SCHEMA.
    # The string-valued field ID intentionally has only numeric values, to
    # ensure that we are enforcing the type at read time rather than
    # trusting to inferred type.  Likewise height.
    INPUT_ROW = (789, 756, 5, False, '2010-11-12', '2018-12-11T11:11:11.222222',
                 '2019-01-23T00:37:46.061780', '999,23')
    EXPECTED_ROW = (789, '756', 5.0, False, datetime.date(2010, 11, 12),
                    datetime.datetime(2018, 12, 11, 11, 11, 11, 222222),
                    datetime.datetime(2019, 1, 23, 0, 37, 46, 61780),
                    (999, 23))  # type: Tuple[PythonType, ...]

    # Set column_to_null to each column in turn, then None (no null columns)
    @data(*([[i] for i in range(len(_TEST_SCHEMA))
             # In recent version of Pandas, boolean types can't be N/A.
             if _TEST_SCHEMA[i].field_type != 'BOOLEAN'] + [[None]]))
    @unpack
    def test_load_table_from_file(self, column_to_null):
        # type: (int) -> None
        dataset_ref = DatasetReference('my_project', 'my_dataset')
        dataset = Dataset(dataset_ref)
        table = Table(TableReference(dataset_ref, 'table1'), _TEST_SCHEMA)
        self.bq_client.create_dataset(dataset)
        self.bq_client.create_table(table)
        output = cStringIO()
        csv_out = csv.writer(output)
        input_row = list(self.INPUT_ROW)
        expected_row = list(self.EXPECTED_ROW)
        if column_to_null is not None:
            input_row[column_to_null] = 'NULL'
            expected_row[column_to_null] = None
        csv_out.writerow(input_row)
        self.bq_client.load_table_from_file(output, table.reference, job_config=None, rewind=True)
        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.table1`',
                                     QueryJobConfig()),
                [expected_row])

    def test_insert_rows(self):
        # type: () -> None
        dataset_ref = DatasetReference('my_project', 'my_dataset')
        dataset = Dataset(dataset_ref)
        table1_ref = TableReference(dataset_ref, 'table1')
        schema = [SchemaField(name="a", field_type='INT64'),
                  SchemaField(name="b", field_type='FLOAT64'),
                  ]
        table = Table(table1_ref, schema)
        self.bq_client.create_dataset(dataset)
        self.bq_client.create_table(table)

        # Insert two rows, check that they landed
        self.assertFalse(self.bq_client.insert_rows(table, [{'a': 1, 'b': 2.5},
                                                            # Intentionally omit 'b' here.
                                                            {'a': 3}]))
        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.table1`',
                                     QueryJobConfig()),
                [[1, 2.5],
                 [3, None]])

        self.assertRowsExpected(
                self.bq_client.query('SELECT a FROM `my_project.my_dataset.table1` WHERE b is NULL',
                                     QueryJobConfig()),
                [[3]])

        # Insert two more rows, check that all four rows are now present.
        self.assertFalse(self.bq_client.insert_rows(table, [{'a': 5, 'b': 6.5},
                                                            {'a': 7, 'b': 8.25}]))
        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.table1`',
                                     QueryJobConfig()),
                [[1, 2.5],
                 [3, None],
                 [5, 6.5],
                 [7, 8.25]])


class ClientWriteFromQueryTest(ClientTestBase):

    def setUp(self):
        super(ClientWriteFromQueryTest, self).setUp()
        dataset_ref = DatasetReference(self.bq_client.project, 'my_dataset')
        schema = [SchemaField(name="a", field_type='INT64'),
                  SchemaField(name="b", field_type='FLOAT64'),
                  ]
        self.source_table = Table(TableReference(dataset_ref, 'source_table'), schema)
        self.destination_table = Table(TableReference(dataset_ref, 'destination_table'), schema)
        self.bq_client.create_dataset(Dataset(dataset_ref))
        self.bq_client.create_table(self.source_table)
        # We don't create the destination table here; some tests do not want it created.

        # Stick two rows into source_table
        self.assertFalse(self.bq_client.insert_rows(self.source_table,
                                                    [{'a': 1, 'b': 2.5}, {'a': 3, 'b': 4.25}]))

    def write_to_table_with_query(self, write_disposition):
        # type: (str) -> None
        """Query all rows from source table, write to destination table w/ requested disposition.

        Args:
            write_disposition: Whether to require the destination table to be empty,
                to append to it, or to overwrite (truncate) it.
        """
        job_config = QueryJobConfig()
        job_config.destination = self.destination_table.reference
        job_config.write_disposition = write_disposition
        self.bq_client.query(
            'SELECT * FROM `{}.{}.{}`'.format(
                    self.source_table.project,
                    self.source_table.dataset_id,
                    self.source_table.table_id), job_config)

    def test_write_query_result_write_disposition_empty(self):
        # type: () -> None
        # You can write into destination_table with WRITE_EMPTY because it's empty
        # Note we do not create the table first; write_empty creates the table.
        self.write_to_table_with_query('WRITE_EMPTY')

        # ... but you can't do that again, because now it's not
        with self.assertRaisesRegexp(ValueError, 'trying to overwrite nonempty table'):
            self.write_to_table_with_query('WRITE_EMPTY')

        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.destination_table`',
                                     QueryJobConfig()),
                [[1, 2.5], [3, 4.25]])

    def test_write_query_result_write_disposition_append(self):
        # type: () -> None

        # You can write into destination_table with WRITE_APPEND
        self.bq_client.create_table(self.destination_table)
        self.write_to_table_with_query('WRITE_APPEND')

        # And you can do it again
        self.write_to_table_with_query('WRITE_APPEND')

        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.destination_table`',
                                     QueryJobConfig()),
                [[1, 2.5], [3, 4.25], [1, 2.5], [3, 4.25]])

    def test_write_query_result_write_disposition_truncate(self):
        # type: () -> None

        self.bq_client.create_table(self.destination_table)
        # Stick a row into destination table
        self.assertFalse(self.bq_client.insert_rows(self.destination_table, [{'a': 5, 'b': 6}]))

        # Overwrite destination_table with the data from source_table with WRITE_TRUNCATE
        self.write_to_table_with_query('WRITE_TRUNCATE')

        self.assertRowsExpected(
                self.bq_client.query('SELECT * FROM `my_project.my_dataset.destination_table`',
                                     QueryJobConfig()),
                [[1, 2.5], [3, 4.25]])


if __name__ == '__main__':
    unittest.main()
