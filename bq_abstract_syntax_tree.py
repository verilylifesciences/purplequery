# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

'''Abstract Syntax Tree.  Each node is an operator or operand.'''
import collections
import uuid
from abc import ABCMeta, abstractmethod
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set,  # noqa: F401
                    Tuple, Union, cast)

import pandas as pd
import six

from bq_types import BQScalarType, BQType, TypedDataFrame, TypedSeries  # noqa: F401

NoneType = type(None)
DatasetType = Dict[str, Dict[str, Dict[str, TypedDataFrame]]]

# column name of ephemeral key added to implement a cross join with pandas merge.
_CROSS_KEY = '__cross_key__'

# Table name for columns that come from evaluating selectors and intermediate expressions.
_SELECTOR_TABLE = '__selector__'


class AbstractSyntaxTreeNode(object):
    '''Base class for AST nodes.'''

    def __repr__(self):
        # type: () -> str
        '''Returns a string representation of this object

        Examples:
          _EmptyNode()
          Value(type_=BQScalarType.STRING, value='something')

        Returns:
            String representation of object
        '''
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(sorted('{}={!r}'.format(key, value)
                             for key, value in six.iteritems(vars(self))
                             if value is not EMPTY_NODE)))

    def strexpr(self):
        # type: () -> str
        '''Return a prefix-expression serialization for testing purposes.'''

        return '({} {})'.format(self.__class__.__name__.upper(),
                                ' '.join(a.strexpr() for a in six.itervalues(vars(self))))

    @classmethod
    def literal(cls):
        # type: () -> Optional[str]
        '''Returns the string that signals the start of an expression.
        For example, a Select class would return "SELECT", while Value would
        return None, because there is no literal that should precede a
        number/string/etc.
        '''
        return None


# These type definitions must live here, after AbstractSyntaxTreeNode is defined
AppliedRuleNode = Union[str,
                        AbstractSyntaxTreeNode,
                        NoneType,
                        Tuple]  # actually Tuple[AppliedRuleNode]; mypy can't do recursive types :(
AppliedRuleOutputType = Tuple[AppliedRuleNode, List[str]]
# Should include Tuple[RuleType] and List[RuleType] but mypy doesn't fully
# support recursive types yet.
RuleType = Union[str, Tuple[Any, ...], List[Any], Callable[[List[str]], AppliedRuleOutputType]]


class MarkerSyntaxTreeNode(AbstractSyntaxTreeNode):
    '''Parent class for abstract syntax tree nodes whose syntax starts with the class name.
    See AbstractSyntaxTreeNode.literal()'''

    @classmethod
    def literal(cls):
        # type: () -> Optional[str]
        return cls.__name__.upper()


class EvaluatableNode(AbstractSyntaxTreeNode):
    '''Abstract base class for syntax tree nodes that can be evaluated to return a column of data'''

    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedDataFrame, TypedSeries]
        '''Generates a new table or column based on applying the instance's
        fields to the given context.

        This method should never be overridden by concrete subclasses.

        Args:
            context: The tables that are involved in this query

        Returns:
            A new table (TypedDataFrame) or column (TypedSeries)
        '''

    @abstractmethod
    def pre_group_by_partially_evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedSeries, EvaluatableNode]
        '''Partially evaluates an expression to prepare for an upcoming GROUP BY.

        This function evaluates expressions that are contained within an aggregation or a grouped by
        field, and stores the results of these in the context to be referred to later.  It is used
        as part of a two-pass evaluation to implement GROUP BY -- see EvaluationContext.do_group_by
        for more detail.

        This function should never be overriden by concrete subclasses.

        Args:
            context: The tables that are involved in this query
        Returns:
            A fully evaluated column or an unevaluated expression (an EvaluatableNode)
        '''

    def name(self):
        # type: () -> Optional[str]
        '''Returns a name for this expression, or None if no name can be inferred.

        Returns:
            The DataFrame returned from a select statement needs a name for every column.  This
            function returns a name that will be used for this expression if the user doesn't
            provide an explicit alias with AS.  If the expression doesn't have an obvious name (most
            expressions), it should return None, and the outermost expression (Selector) will
            generate one.
        '''
        return None

    @abstractmethod
    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        '''Returns a new syntax tree rooted at the current one, marking fields that are grouped by.

        Args:
            group_by_paths: Canonicalized paths of the columns that are grouped by
            context: Context to evaluate in (for canonicalizing)
        '''


class EvaluatableLeafNode(EvaluatableNode):
    '''Abstract Syntax Tree Node that can be evaluated and has no child nodes.'''

    def evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedSeries, TypedDataFrame]
        '''See docstring in EvaluatableNode.evaluate'''
        return self._evaluate_leaf_node(context)

    def pre_group_by_partially_evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedSeries, EvaluatableNode]
        '''See docstring in EvaluatableNode.pre_group_by_partially_evaluate'''
        result = self._evaluate_leaf_node(context)
        if isinstance(result, TypedDataFrame):
            # TODO: This codepath can be hit in a query like SELECT * GROUP BY
            # Such a query can actually be valid if every *-selected field is
            # grouped by.  Support this.
            raise ValueError("Cannot partially evaluate {!r}".format(self))
        return result

    @abstractmethod
    def _evaluate_leaf_node(self, context):
        # type: (EvaluationContext) -> Union[TypedDataFrame, TypedSeries]
        '''Computes a new column from this childless node in the provided context.

        This method must be overriden by all subclasses.

        Args:
            context: EvaluationContext to evaluate this expression in

        Returns:
            A table or column of data.
        '''


class EvaluatableNodeWithChildren(EvaluatableNode):
    '''Abstract Syntax Tree node that can be evaluated, based on child nodes.'''

    def __init__(self, children):
        # type: (Sequence[EvaluatableNode]) -> None
        self.children = children

    def pre_group_by_partially_evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedSeries, EvaluatableNode]
        '''See docstring in EvaluatableNode.pre_group_by_partially_evaluate'''
        evaluated_children = [child.pre_group_by_partially_evaluate(context)
                              for child in self.children]

        # Expressions whose children are fully evaluated can be fully evaluated themselves.
        # For example, these expressions will fully evaluate into a column of numbers (a
        # TypedSeries).
        # - a * 2
        # - concat(foo, "/", bar)
        if all(isinstance(child, TypedSeries) for child in evaluated_children):
            return self._evaluate_node(cast(List[TypedSeries], evaluated_children))

        # Expressions whose children are not fully evaluated should not keep the caches of partial
        # evaluation; not being fully evaluated means that this expression is outside of
        # aggregation, and so will be evaluated in the second pass.
        # For example, consider this expression: 2 + max(b * c).
        # pre_group_by_partially_evaluate can't evaluate the 'max' function yet, because that needs
        # to happen after the group_by.  So, the 'max' will defer evaluation and return an
        # EvaluatableNode (see EvaluatableNodeThatAggregatesOrGroups.pre_group_by_partially_evaluate
        # below), but the '2' will return an actual column of twos (a TypedSeries).  We don't
        # want those twos to get grouped, because we need to add them column to the result of
        # max and we can't add a SeriesGroupBy to a Series.  So, we ignore the evaluation result of
        # any column that is fully evaluated, and just keep the original column node instead.
        else:
            return self.copy([
                evaluated_child if isinstance(evaluated_child, EvaluatableNode) else child
                for child, evaluated_child in zip(self.children, evaluated_children)])

    def _ensure_fully_evaluated(self, evaluated_children):
        # type: (List[Any]) -> List[TypedSeries]
        '''Ensure evaluated_children are fully evaluated; raise ValueError if not.'''
        if not all(isinstance(child, TypedSeries) for child in evaluated_children):
            raise ValueError(
                    "In order to evaluate {}, all children must be evaluated.".format(self))
        return cast(List[TypedSeries], evaluated_children)

    def evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedDataFrame, TypedSeries]
        '''Generates a new table or column based on applying the instance's
        fields to the given context.

        This method should never be overridden by subclasses.

        Args:
            context: The tables that are involved in this query

        Returns:
            A new table (TypedDataFrame) or column (TypedSeries)
        '''
        evaluated_children = [child.evaluate(context) for child in self.children]
        return self._evaluate_node(self._ensure_fully_evaluated(evaluated_children))

    @abstractmethod
    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        '''Computes a new column based on the evaluated arguments to this expression.

        This method must be overriden by all subclasses.

        Args:
            evaluated_children: The already-evaluated children of this node.
        Returns:
            A new column (TypedSeries)
        '''

    @abstractmethod
    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> EvaluatableNode
        '''Creates a new version of the current object with new, different children

        EvaluatableNode subclasses have two kinds of state - the children, the nodes below this one
        in the tree, and other state, like the specific binary operator or the particular function
        that a node invokes.  This method creates a new object that changes out the children
        for different nodes, but keeps the other state, allowing for operations that rewrite the
        abstract syntax tree.

        This method must be overriden by all subclasses.

        Args:
            new_children: Abstract syntax tree nodes to use as the children of the new node.
        Returns:
            A new node.
        '''

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        '''Returns a new syntax tree rooted at the current one, marking fields that are grouped by.

        Args:
            group_by_paths: Canonicalized paths of the columns that are grouped by
            context: Context to evaluate in (for canonicalizing)
        '''
        return self.copy([child.mark_grouped_by(group_by_paths, context)
                          for child in self.children])


class EvaluatableNodeThatAggregatesOrGroups(EvaluatableNodeWithChildren):
    '''Abstract Syntax Tree node that can be evaluated that aggregates child nodes.

    Subclasses represent expressions like count, max, or sum that aggregate results of multiple
    rows into one row.

    GroupBy is also a subclass; this expression node wraps Fields that are listed in GROUP BY and
    are not themselves aggregated.
    '''

    def pre_group_by_partially_evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedSeries, EvaluatableNode]
        '''See docstring in EvaluatableNode.pre_group_by_partially_evaluate'''
        # aggregating and grouped by expressions must fully evaluate children, then we cache
        # those expressions into the context and return a new AST node that computes the aggregate
        # over lookups from the cache (which will be grouped in the second-pass evaluation).
        evaluated_children = self._ensure_fully_evaluated(
            [child.pre_group_by_partially_evaluate(context)
             for child in self.children])
        return self.copy(
                [Field(context.maybe_add_column(evaluated_child))
                 for evaluated_child in evaluated_children])

    def evaluate(self, context):
        # type: (EvaluationContext) -> Union[TypedDataFrame, TypedSeries]
        '''See docstring in EvaluatableNode.evaluate.'''
        evaluated_children = self._ensure_fully_evaluated(
            [child.evaluate(context) for child in self.children])
        if context.group_by_paths:
            return self._evaluate_node_in_group_by(evaluated_children)
        else:
            return self._evaluate_node(evaluated_children)

    @abstractmethod
    def _evaluate_node_in_group_by(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        '''Computes a new column based on the evaluated arguments in a group by context.

        Unlike _evaluate_node, here the evaluated children are grouped by some set of fields;
        this function evalutes this expression on those arguments.

        This method must be overriden by all subclasses.

        Args:
            evaluated_children: The already-evaluated children of this node.
        Returns:
            A new column (TypedSeries)
        '''

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        '''Returns a new syntax tree rooted at the current one, marking fields that are grouped by.

        Args:
            group_by_paths: Canonicalized paths of the columns that are grouped by
            context: Context to evaluate in (for canonicalizing)
        '''
        # If columns that are grouped by appear inside of an aggregating expression, they will
        # be aggregated by that expression, so we should not mark them as grouped by.
        return self


class TableContext(object):
    '''Context for resolving a name or path to a table (TypedDataFrame).

    Typically applied in a FROM statement.

    Contrast with EvaluationContext, whose purpose is to resolve a name to a column (TypedSeries).
    '''

    def lookup(self, path):
        # type: (Tuple[str, ...]) -> Tuple[TypedDataFrame, Optional[str]]
        '''Look up a path to a table in this context.

        Args:
            path: A tuple of strings representing a period-separated path to a table, like
                projectname.datasetname.tablename, or just tablename

        Returns:
            The table of data (TypedDataframe) found, and its name.
        '''
        raise KeyError("Cannot resolve table `{}`".format('.'.join(path)))


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
        # type: (Tuple[str, ...]) -> Tuple[TypedDataFrame, Optional[str]]
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
                path = (project, dataset) + path
            else:
                # Dataset and table both specified
                path = (project,) + path

        if len(path) > 3:
            raise ValueError("Invalid path has more than three parts: {}".format(path))
        project_id, dataset_id, table_id = path

        return self.datasets[project_id][dataset_id][table_id], table_id


class DataframeNode(AbstractSyntaxTreeNode):
    '''Abstract Syntax Tree Nodes that have a get_dataframe() method.

    This node represents a syntactic object that can be selected FROM or that
    can be JOINed to another DataframeNode.
    '''

    def get_dataframe(self, table_context, outer_context=None):
        # type: (TableContext, Optional[EvaluationContext]) -> Tuple[TypedDataFrame, Optional[str]]
        '''Scope the given datasets by the criteria specified in the
        instance's fields.

        Args:
            datasets: All the tables in the database
            outer_context: Context of a containing query (e.g. an EXISTS expression)

        Returns:
            Tuple of the resulting table (TypedDataFrame) and a name for
            this table
        '''


class GroupedBy(EvaluatableNodeThatAggregatesOrGroups):
    '''One of the columns grouped by.

    When GROUP BY is used, every column that is SELECTed must either be aggregated (min, max, etc)
    or be one of the columns grouped by, otherwise it might have multiple values across the group,
    which doesn't make sense to select one of.  In this implementation, this manifests as follows:
    when a column is evaluated, the expressions are pandas SeriesGroupBy objects, and those need to
    be either aggregated into Serieses, or marked with a GroupedBy node as being one of the columns
    grouped by.  A GroupedBy node's child will evaluate to an expression constant within its group
    (by definition) but we need to explicitly convert it to a Series containing those constant
    elements.
    '''

    def __init__(self, expression):
        self.children = [expression]

    def copy(self, new_children):
        # type: (Sequence[EvaluatableNode]) -> GroupedBy
        return GroupedBy(new_children[0])

    def _evaluate_node(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        raise ValueError(
                "It does not make sense to evaluate GroupedBy outside of a grouped context.")

    def _evaluate_node_in_group_by(self, evaluated_children):
        # type: (List[TypedSeries]) -> TypedSeries
        evaluated_expression, = evaluated_children
        if not evaluated_expression.series.min().equals(evaluated_expression.series.max()):
            raise ValueError("Field {} should be constant within group but it varies {}"
                             .format(self, evaluated_expression))
        return TypedSeries(evaluated_expression.series.min(), evaluated_expression.type_)


class Field(EvaluatableLeafNode):
    '''A reference to a column in a table.

    For example, in a table Table with columns A, B, and C,
    a valid Field path would be ('B',) or ('Table', 'A').
    '''
    def __init__(self, path):
        # type: (Tuple[str, ...]) -> None
        '''Set up Field node

        Args:
            path: A tuple of strings describing the path to this field
                (just the column name, or table.column)
        '''
        self.path = path

    def strexpr(self):
        # type: () -> str
        return '.'.join(self.path)

    def mark_grouped_by(self, group_by_paths, context):
        # type: (Sequence[Tuple[str, ...]], EvaluationContext) -> EvaluatableNode
        if context.get_canonical_path(self.path) in group_by_paths:
            return GroupedBy(self)
        return self

    def name(self):
        # type: () -> Optional[str]
        return self.path[-1]

    def _evaluate_leaf_node(self, context):
        # type: (EvaluationContext) -> TypedSeries
        # Retrieves the column from the context
        result = context.lookup(self.path)
        result.series.name = self.name()
        return result

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, Field):
            return self.path == other.path
        return False


class _EmptyNode(AbstractSyntaxTreeNode):
    '''An Empty syntax tree node; represents an optional element not present.'''

    def strexpr(self):
        # type: () -> str
        return 'null'


EMPTY_NODE = _EmptyNode()


class EvaluationContext:
    '''Context for resolving a name to a column (TypedSeries).

    An EvaluationContext is a representation of columns in all the tables that
    are in the query's scope.  Typically used in the context of SELECT, WHERE, etc.

    Contrast with TableContext, whose purpose is to resolve a name to a table (TypedDataFrame),
    and contains all the tables (all the data) that is available in the database.
    '''

    def __init__(self, table_context):
        # type: (TableContext) -> None
        '''Initialize a context.

        Args:
            table_context: All tables visible to be queried.
        '''

        # We don't want an actual empty dataframe because with no FROMed tables, we
        # still want to return a single row of results.
        self.table = TypedDataFrame(pd.DataFrame([[1]]), [None])  # type: TypedDataFrame

        # Mapping of column ID (prefixed by table ID) to its type
        self.canonical_column_to_type = {}  # type: Dict[str, BQType]

        # Mapping of column IDs to list of table IDs to which they belong
        self.column_to_table_ids = collections.defaultdict(list)  # type: Dict[str, List[str]]

        # All the available datasets (not necessarily all present in this context, yet)
        self.table_context = table_context

        # Table ids included in this context.
        self.table_ids = set()  # type: Set[str]

        # Stores the list of columns grouped by, or None if this expression isn't grouped.
        self.group_by_paths = None  # type: Optional[List[Tuple[str, str]]]

        # Additional context (names of variables) used for looking up fields, but not for
        # grouping.
        self.subcontext = None  # type: Optional[EvaluationContext]

        # If true, then aggregation expressions will not be evaluated, but instead will
        # return a syntax tree node, corresponding to their expression with any subexpressions
        # properly evaluated.
        self.exclude_aggregation = False

        # The names of the selector columns.  Used to canonicalize references to selectors to
        # a path (_SELECTOR_TABLE, name)
        self.selector_names = []  # type: List[str]

    def add_subcontext(self, subcontext):
        # type: (EvaluationContext) -> None
        '''Adds another context to this one.

        The subcontext will be used for looking up columns.  It's necessary when the subcontext
        is a group by context, but the current context is not, as we can't join a pandas
        DataFrame and DataFrameGroupBy into one object.

        Args:
            subcontext: An EvaluationContext to look up fields in, disjunct from the current one.
        '''
        if self.subcontext is not None:
            raise ValueError("Context already has subcontext {}!".format(self.subcontext))
        self.subcontext = subcontext

    def maybe_add_column(self, column):
        # type: (TypedSeries) -> Tuple[str, ...]
        '''Ensures that column is available in the context.  Returns the path to retrieve it later.

        Calling this function indicates that column is an evaluated series of data that should be
        available in the evaluation context.  Either find it in the existing context, or add it as
        a new column to the context.  Then, return the identifier path to retrieve it later out of
        the context, either the path to the existing column, or a new one.  This may involve
        generating a random name for the column if it doesn't contain one.

        Args:
            column: A TypedSeries of data, optionally with a name.

        Returns:
            An identifier path to where this column can be found in the context.
        '''
        name = column.series.name or uuid.uuid4().hex

        canonical_paths = self.get_all_canonical_paths((name,))
        if len(canonical_paths) > 1:
            raise ValueError("It's confusing to add column {}; ambiguous {}".format(
                    name, canonical_paths))
        if canonical_paths:
            canonical_path, = canonical_paths
        else:
            canonical_path = (_SELECTOR_TABLE, name)
        if canonical_path[0] != _SELECTOR_TABLE:
            # We already know about this name; don't need to add to context.
            return canonical_paths[0]
        self.column_to_table_ids[name] = [canonical_path[0]]
        column.series.name = '.'.join(canonical_path)
        self.canonical_column_to_type[column.series.name] = column.type_
        self.table = TypedDataFrame(
                pd.concat([self.table.dataframe, column.series], axis=1),
                self.table.types + [column.type_])
        return canonical_path

    def do_group_by(self, selectors, group_by):
        # type: (Sequence[EvaluatableNode], List[Field]) -> List[EvaluatableNode]
        """Groups the current context by the requested paths.

        Canonicalizes the paths (figures out which table a plain column name goes with), applies
        the pandas groupby operation to the context's table, and saves the group by paths to mark
        this context as a Group By context (which changes how aggregating expressions like sum
        or max operate, from operating over all rows to just the rows within a group).

        Args:
            paths: A list of column paths, i.e. a column name or a table, column pair.
                These are as requested by the user string.
        """
        if isinstance(self.table.dataframe, pd.core.groupby.DataFrameGroupBy):
            raise ValueError("Context already grouped!")
        group_by_paths = [self.get_canonical_path(field.path) for field in group_by]
        marked_selectors = [field.mark_grouped_by(group_by_paths, self) for field in selectors]
        partially_evaluated_selectors = [
            field.pre_group_by_partially_evaluate(self) for field in marked_selectors]
        new_selectors = [Field(self.maybe_add_column(selector)) if isinstance(selector, TypedSeries)
                         else selector for selector in partially_evaluated_selectors]

        group_by_fields = ['.'.join(path) for path in group_by_paths]
        grouped = self.table.dataframe.groupby(by=group_by_fields)
        self.table = TypedDataFrame(grouped, self.table.types)
        self.group_by_paths = group_by_paths
        return new_selectors

    def add_table_from_node(self, from_item, alias):
        # type: (DataframeNode, Union[_EmptyNode, str]) -> Tuple[TypedDataFrame, str]
        '''Add a table to the query's scope when it is FROM-ed or JOIN-ed.

        Args:
            from_item: A node representing a FROM expression
            alias: An alias for the given table
        Returns:
            Tuple of the table that has been added to the scope
            (TypedDataFrame) and its name/label.
        '''

        table, table_id = from_item.get_dataframe(self.table_context)
        return self.add_table_from_dataframe(table, table_id, alias)

    def add_table_from_dataframe(self, table,  # type: TypedDataFrame
                                 table_id,  # type: Optional[str]
                                 alias  # type: Union[_EmptyNode, str]
                                 ):
        # type: (...) -> Tuple[TypedDataFrame, str]
        '''Add a table to the query's scope, given an already resolved DataFrame.

        Args:
            dataframe: The table to add, already in TypedDataFrame format
            table_id: The default table name, if available
            alias: An alias for the given table
        Returns:
            Tuple of the table that has been added to the scope
            (TypedDataFrame) and its name/label.
        '''
        # Alias, if provided, defines the table's name
        if not isinstance(alias, _EmptyNode):
            table_id = alias
        elif not table_id:
            table_id = '__join{}'.format(len(self.table_ids))

        # Save mapping of column ID to table IDs
        for column in table.dataframe.columns:
            column_name = column.split('.')[-1]
            self.column_to_table_ids[column_name].append(table_id)

        self.table_ids.add(table_id)

        # Rename columns in format "[new_table_id].[column_name]"
        table = TypedDataFrame(
            table.dataframe.rename(
                columns=lambda column_name: '{}.{}'.format(table_id, column_name.split('.')[-1])),
            table.types)

        # Save mapping of column ID to type
        if len(table.dataframe.columns) != len(table.types):
            raise ValueError('Context: Number of columns and types not equal: {} != {}'.format(
                len(table.dataframe.columns), len(table.types)))
        for column, type_ in zip(table.dataframe.columns, table.types):
            self.canonical_column_to_type[column] = type_

        self.table = table
        return table, table_id

    @classmethod
    def clone_context_new_table(cls, table, old_context):
        # type: (TypedDataFrame, EvaluationContext) -> EvaluationContext
        '''Clone a context - use all the same metadata as the given ("old")
        context except for the table, which is specified separately.
        The new table must contain the same columns as the old context's table
        in order for the metadata to remain valid.

        Args:
            table: TypedDataFrame containing the data that the new context represents
            old_context: Old context from which to copy metadata
        '''
        if any([new != old for (new, old) in zip(table.dataframe.columns,
                                                 old_context.table.dataframe.columns)]):
            raise ValueError('Columns differ when cloning context with new table: {} vs {}'.format(
                table.dataframe.columns, old_context.table.dataframe.columns))
        new_context = cls(old_context.table_context)
        new_context.table = table
        new_context.canonical_column_to_type = old_context.canonical_column_to_type
        new_context.column_to_table_ids = old_context.column_to_table_ids
        new_context.table_ids = old_context.table_ids
        new_context.group_by_paths = old_context.group_by_paths
        new_context.subcontext = old_context.subcontext
        new_context.exclude_aggregation = old_context.exclude_aggregation
        return new_context

    def get_all_canonical_paths(self, path):
        # type: (Tuple[str, ...]) -> List[Tuple[str, str]]
        '''
        Find all the possible table-column pairs that the given path could refer to.

        Args:
            path: A tuple of either just (column name) or (table name, column name)
        Returns:
            A list of (table name, column name) tuples
        '''
        result = []  # type: List[Tuple[str, str]]
        if len(path) == 1:
            column, = path
            result = [(table_id, column) for table_id in self.column_to_table_ids[column]]
        elif len(path) == 2:
            # Check that this path is valid
            table_id, column = path
            if table_id not in self.column_to_table_ids[column] and not self.subcontext:
                raise ValueError("field {} is not present in table {} (only {})".format(
                        column, table_id, self.column_to_table_ids[column]))
            result = [(table_id, column)]
        else:
            raise NotImplementedError('Array fields are not implemented; path {}'.format(path))

        # If path wasn't found in the current context, try the subcontext
        if not result and self.subcontext:
            result = self.subcontext.get_all_canonical_paths(path)
        if not result and column in self.selector_names:
            return [(_SELECTOR_TABLE, path[0])]
        return result

    def get_canonical_path(self, path):
        # type: (Tuple[str, ...]) -> Tuple[str, str]
        '''
        Get exactly one table-column pair for the given path (or throw error).
        If path is already a (table, column) tuple, this just makes sure it exists in the dataset.
        If table name is not provided in the path, this finds the table that the column belongs to.

        Args:
            path: A tuple of either just (column name) or (table name, column name)
        Returns:
            A tuple of the table and column name
        '''
        all_paths = self.get_all_canonical_paths(path)
        field = '.'.join(path)
        if len(all_paths) == 0:
            raise ValueError("field {} is not present in any from'ed tables".format(field))
        if len(all_paths) > 1:
            raise ValueError("field {} is ambiguous: present in {!r}".format(field, all_paths))
        return all_paths[0]

    def lookup(self, path):
        # type: (Tuple[str, ...]) -> TypedSeries
        '''
        Gets a column from the context's table.

        Args:
            path: Path to the column, as a tuple of strings
        Returns:
            A TypedSeries representing the requested column, or a KeyError if not found
        '''
        key = '.'.join(self.get_canonical_path(path))

        # First, try looking up the path in the current context.  If the path is not found,
        # it's not an error (yet), as we continue on and look in the subcontext, if one exists.
        try:
            series = self.table.dataframe[key]
        except KeyError:
            if self.subcontext:
                return self.subcontext.lookup(path)
            else:
                raise KeyError(("path {!r} (canonicalized to key {!r}) not present in table; "
                                "columns available: {!r}").format(path, key,
                                                                  list(self.table.dataframe)))
        try:
            type_ = self.canonical_column_to_type[key]
        except KeyError:
            raise KeyError(("path {!r} (canonicalized to key {!r}) not present in type dict; "
                            "columns available: {!r}").format(
                                    path, key, list(self.canonical_column_to_type.keys())))
        return TypedSeries(series, type_)


EMPTY_CONTEXT = EvaluationContext(TableContext())
