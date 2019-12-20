# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Defines classes for storing data for the BigQuery fake."""

from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set,  # noqa: F401
                    Tuple, Union, cast)

from .bq_types import BQScalarType, BQType, TypedDataFrame, TypedSeries  # noqa: F401

NoneType = type(None)
DatasetType = Dict[str, Dict[str, Dict[str, TypedDataFrame]]]

# Table name for columns that come from evaluating selectors and intermediate expressions.
_SELECTOR_TABLE = '__selector__'


class TableContext(object):
    '''Context for resolving a name or path to a table (TypedDataFrame).

    Typically applied in a FROM statement.

    Contrast with EvaluationContext, whose purpose is to resolve a name to a column (TypedSeries).
    '''

    def lookup(self, path):
        # type: (Sequence[str]) -> Tuple[TypedDataFrame, Optional[str]]
        '''Look up a path to a table in this context.

        Args:
            path: A sequence of strings representing a period-separated path to a table, like
                projectname.datasetname.tablename, or just tablename

        Returns:
            The table of data (TypedDataframe) found, and its name.
        '''
        raise KeyError("Cannot resolve table `{}`".format('.'.join(path)))

    def set(self, path, table):
        # type: (Tuple[str, ...], TypedDataFrame) -> None
        '''Sets a path to refer to a table.

        Args:
            path: A tuple of strings representing a period-separated path to a table, like
                projectname.datasetname.tablename, or just tablename
            table: The table to add
        '''
        raise NotImplementedError("Abstract method, not implemented")


class DatasetTableContext(TableContext):
    '''A TableContext containing a set of datasets.'''

    def __init__(self, datasets):
        # type: (DatasetType) -> None
        '''Construct the TableContext.

        Args:
            datasets: A series of nested dictionaries mapping to a TypedDataFrame.
            For example, {'my_project': {'my_dataset': {'table1': t1, 'table2': t2}}},
            where t1 and t2 are two-dimensional TypeDataFrames representing a table.
        '''
        self.datasets = datasets

    def lookup(self, path):
        # type: (Sequence[str]) -> Tuple[TypedDataFrame, Optional[str]]
        '''See TableContext.lookup for docstring.'''
        if not self.datasets:
            raise ValueError("Attempt to look up path {} with no projects/datasets/tables given."
                             .format(path))
        if len(path) < 3:
            # Table not fully qualified - attempt to resolve
            if len(self.datasets) != 1:
                raise ValueError("Non-fully-qualified table {} with multiple possible projects {}"
                                 .format(path, sorted(self.datasets.keys())))
            project, = self.datasets.keys()

            if len(path) == 1:
                # No dataset specified, only table
                if len(self.datasets[project]) != 1:
                    raise ValueError(
                            "Non-fully-qualified table {} with multiple possible datasets {}"
                            .format(path, sorted(self.datasets[project].keys())))
                dataset, = self.datasets[project].keys()
                path = (project, dataset) + tuple(path)
            else:
                # Dataset and table both specified
                path = (project,) + tuple(path)

        if len(path) > 3:
            raise ValueError("Invalid path has more than three parts: {}".format(path))
        project_id, dataset_id, table_id = path

        return self.datasets[project_id][dataset_id][table_id], table_id

    def set(self, path, table):
        # type: (Tuple[str, ...], TypedDataFrame) -> None
        '''Sets a path to refer to a table.

        Args:
            path: A tuple of strings representing a period-separated path to a table, like
                projectname.datasetname.tablename
            table: The table to add
         '''
        project_id, dataset_id, table_id = path
        if project_id not in self.datasets:
            raise ValueError("Attempting to create {!r} but project {!r} not created"
                             .format(path, project_id))
        self.datasets[project_id].setdefault(dataset_id, {})[table_id] = table
