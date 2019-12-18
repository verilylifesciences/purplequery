# PurpleQuery

## Overview

PurpleQuery is a partial implementation of BigQuery for testing.  All
table creation and querying happens in-memory.  The parser is a straightforward
translation of the published BigQuery syntax, and the backend is based on
Pandas.  The provided API client is a drop-in replacement for the Google Python
BigQuery client.

PurpleQuery also integrates smoothly with Verily's [analysis-py-utils BigQuery
wrapper](https://github.com/verilylifesciences/analysis-py-utils/tree/master/verily/bigquery_wrapper),
which provides a simplified API for interacting with BigQuery.

PurpleQuery is compatible with Python 2 and 3.

## Installation

```sh
pip install git+https://github.com/verilylifesciences/purplequery.git
```

## Supported Features
PurpleQuery supports
- WITH ... SELECT [DISTINCT] [EXCEPT] [REPLACE] ... FROM ... JOIN ... WHERE ... GROUP BY ... ORDER BY ... LIMIT ...
- INNER, OUTER, CROSS joins
- UNION ALL
- Analytic functions
- Arithmetic expressions

Functions including
- ARRAY
- ARRAY_AGG
- CASE
- CAST
- CONCAT
- COUNT
- EXISTS
- EXTRACT
- IF
- (NOT) IN
- IS (NOT) NULL
- MAX
- MIN
- MOD
- NOT
- ROW\_NUMBER
- STRUCT
- SUM
- TIMESTAMP

## Unsupported features
- UNNEST columns (UNNEST array expressions are supported)
- many functions
- set operations besides UNION ALL
- UPDATE and other mutation operations in queries (these are supported via the
  Python API)
- Window identifiers and window frames for analytic function calls

## Usage

```python

from google.cloud.bigquery import Dataset, DatasetReference, Table, TableReference
from google.cloud.bigquery.job import QueryJobConfig
from google.cloud.bigquery.schema import SchemaField
from purplequery import Client

bq_client = Client('my_project')

dataset_ref = DatasetReference('my_project', 'dataset1')
bq_client.create_dataset(Dataset(dataset_ref))
table = Table(TableReference(dataset_ref, 'table1'),
              [SchemaField(name='num', field_type='INTEGER'),
               SchemaField(name='ID', field_type='STRING')])
bq_client.create_table(table)
bq_client.insert_rows(table, [{'num': 1, 'ID': 'first'},
                              {'num': 2, 'ID': 'second'}])
job = bq_client.query('SELECT * FROM `my_project.dataset1.table1`',
                      QueryJobConfig())
rows = [list(row.values()) for row in job.result()]

print rows
```

```
[[1, 'first'], [2, 'second']]
```

## Usage with bigquery\_wrapper

To use with the Verily BigQuery wrapper, use the `alternate_bq_client_class`
parameter to use PurpleQuery instead, and the `self.bq_client` object available
to test methods will use PurpleQuery instead of the Google BigQuery client.


```python
from verily.bigquery_wrapper import bq_test_case
from purplequery import Client as FakeClient

class MyTest(bq_test_case.BQTestCase):

    @classmethod
    def setUpClass(cls, use_mocks=False):
        super(Mytest, cls).setUpClass(
            use_mocks=use_mocks,
            alternate_bq_client_class=FakeClient)

    ...
```

## License

See [the license](LICENSE)

## Contributing

See [instructions for contributing to this project](CONTRIBUTING.md)

## Troubleshooting

PurpleQuery is not a complete implementation of BigQuery.  For (rare) cases
where an SQL statement is not understood by PurpleQuery's parser, the result
will be an exception `Could not fully parse query`, followed by the tokens
following the section that could be parsed.  Typically the beginning of this
section will be the syntactic element that PurpleQuery does not understand.
See `grammar.py` for the grammar understood; feel free to extend it!

`NotImplementedError` is raised when a syntactic element is parsed but is
not implemented in the backend, e.g. a function.  See `evaluatable_node.py` to
add a new function or other expression element.

## Running tests

Clone the repository:
```
git clone https://github.com/verilylifesciences/purplequery.git
```

Build the docker container:
```
docker build -t purplequery purplequery/
```

Run the tests in the docker container:
```
docker run --rm -ti -v `pwd`/purplequery:/pq -w /pq purplequery /pq/run_tests.sh
```
