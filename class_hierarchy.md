# Class Hierarchy

[TOC]

This page describes the Python class hierarchy in fake BigQuery.

## Abstract Syntax Tree nodes
The output of the parser is an abstract syntax tree.  The nodes in this tree are
subclasses of AbstractSyntaxTreeNode.

```
AbstractSyntaxTreeNode                     # Abstract base class of all syntax tree nodes.
|\
| MarkerSyntaxTreeNode                     # Parent of nodes that start with their class: Select, If,...
|\
| EmptyNode                                # Represents a optional syntactic node not present.
|\
| DataSource                               # Represents what comes after FROM; all tables JOINed
|\
  StarSelector                             # The * in SELECT * [EXCEPT] [REPLACE]
|\
| DataFrameNode                            # Base class of nodes that compute a whole table
| |\
| | QueryExpression                        # Outermost SQL statement; WITH ... ORDER BY ...  LIMIT ...
| |\
| | Select                                 # A SELECT statement
| |\
| | SetOperation                           # A combination of two query expressions by a set operation such as UNION ALL
|  \
|   TableReference                         # A reference to a table: [[project.]dataset.]table
|  \
|   Unnest                                 # An UNNEST(array-expression)
 \
  EvaluatableNode                          # Abstract: a node that can be evaluated; an expression
  |\
  | EvaluatableLeafNode                    # Abstract: An expression without any references to other EvaluatableNodes
  | |\
  | | Field                                # A reference to a single column; table.a
  | |\
  | | Value                                # A literal constant; 3.4 or 5 or "hello"
  |  \
  |   Exists                               # EXISTS (SELECT ...)
  |\
  | EvaluatableNodeThatAggregatesOrGroups  # Abstract
  | |\
  | | GroupedBy                            # A wrapper around a Field that's in the GROUP BY list
  |  \
  |   AggregatingFunctionCall              # A call to a function that aggregates, sum, max, ...
   \
    EvaluatableNodeWithChildren            # Abstract
    |\
    | Array                                # An array literal; ARRAY<type>[expr, ...]
    |\
    | BinaryExpression                     # A binary expression, 3 + 4, a < b, etc.
    |\
    | Case                                 # CASE ... WHEN ... ELSE
    |\
    | Cast                                 # CAST (... AS type)
    |\
    | Extract                              # EXTRACT (date_part FROM date)
    |\
    | If                                   # IF(condition, then, else)
    |\
    | InCheck                              # foo [NOT] IN (...)
    |\
    | Not                                  # NOT expression
    |\
    | NullCheck                            # foo IS [NOT] NULL
    |\
    | Selector                             # Wrapper around the outermost expressions in a Select
    |\
    | Struct                               # A STRUCT expression
    |\
    | UnaryNegation                        # -expression
    |\
    | NonAggregatingFunctionCall           # A call to a function that doesn't aggregate: concat, timestamp, ...
     \
      AnalyticFunctionCall                 # A call to a function that evaluates over a window: sum over, row_number over, ...
```

## Functions and Function calls
Functions and function calls are represented by two inheritance trees.  Function
_calls_, i.e. the expression indicating that a function is invoked, are
descendants of FunctionCall and also of EvaluatableNode.

### Function calls
```
FunctionCall                 # Base class of function call expressiopns
|\
| AggregatingFunctionCall    # Expression aggregating many rows into one row
|\
| AnalyticFunctionCall       # Expression evaluating windows of rows, one value per row
 \
  NonAggregatingFunctionCall # Expression evaluated row by row
```

Functions are descendants of the Function class, which represents the function
independent of a particular evaluation.  Functions may be invoked as an
aggregating, non-aggregating, or analytic function call, depending on the syntax
and the type of function.

### Functions
```
Function                  # Abstract base class of all functions
|\
| AggregatingFunction     # Function that aggregates many values into one value
| |\
| | Array_agg             # Aggregate a column into an array-valued cell
| |\
| | Count                 # Count of nonempty values.
| |\
| | Max                   # Maximum value
| |\
| | Min                   # Minimum value
|  \
|   Sum                   # Sum of values
 \
  NonAggregatingFunction  # Function that computes one output per row
  |\
  | Concat                # Concatenate strings
  |\
  | Current_Timestamp     # Current timestamp (i.e. now).
  |\
  | Mod                   # Modulus, aka remainder after division
  |\
  | Row_number            # (analytic) Numbers the rows in a column.
   \
    Timestamp             # Convert to timestamp
```

## Types

BigQuery types are explicitly represented in the fake so that they can be
accurately transmitted back to the caller.

```
BQType           # Base class of all types
|\
| BQScalarType   # A scalar type - integer, float, etc.
 \
  BQArray        # Array (aka Repeated) type - an array of scalars.
```

In addition, these two types annotate Pandas types with the BigQuery type stored
in them.

- TypedSeries: a column of data, and its type
- TypedDataFrame: a table of data, and its types.

## Context classes.

Context classes are used to resolve identifiers during evaluation; i.e. to turn
a user-specified name into an actual column or table of data.

### Table contexts

```
TableContext           # Base class of contexts that resolve references to tables
|\
| DatasetTableContext  # Table context based on a dictionary of project/dataset/table names.
 \
  WithTableContext     # Table context based on tables introduced by WITH statements.
```

### Column context

`EvaluationContext` represents the data needed to evaluate expressions.  To turn
  'select a+b, c from table1, table2' into an actual table of numbers,
  an EvaluationContext holds the actual columns of data corresponding to a, b,
  and c.
