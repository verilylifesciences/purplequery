"""Grammar for BigQuery types."""

from typing import List, Sequence, Tuple, Union  # noqa: F401

from .bq_abstract_syntax_tree import AppliedRuleOutputType  # noqa: F401
from .bq_types import BQType  # noqa: F401
from .bq_types import BQArray, BQScalarType, BQStructType
from .query_helper import apply_rule, separated_sequence, wrap
from .terminals import identifier


def bigquery_type(tokens):
    # type: (List[str]) -> AppliedRuleOutputType
    """Grammar rule recognizing a BigQuery type.

    This is written as a function because it's recursive with the array and struct type rules.

    Args:
        tokens: Parts of the user's query (split by spaces into tokens) that
        are not yet parsed
    Returns:
        A tuple of the function call node representing the result of applying the rule
        to the tokens, and the remaining unmatched tokens, or None and all the tokens if
        the rule does not match.
    """
    return apply_rule([array_type, struct_type, scalar_type], tokens)


def _create_struct_type(typed_fields):
    # type: (Sequence[Union[Tuple[str, BQType], BQType]]) -> BQStructType
    '''Create a BQStructType instance from the output of the grammar rule.

    Args:
        typed_fields: A sequence of field specifications.  Each field can be specified with
           either just a type (in which case the name is unspecified), or a pair of a name and a
           type.
    Returns:
        The BQStructType object representing the user-specified type.
    '''
    fields = []
    types = []
    for elt in typed_fields:
        if isinstance(elt, tuple):
            field, type_ = elt
            fields.append(field)
            types.append(type_)
        else:
            fields.append(None)
            types.append(elt)
    return BQStructType(fields, types)


# Grammar rule for a scalar type (INTEGER, FLOAT, etc)
scalar_type = wrap(BQScalarType.from_string, identifier)

# Grammar rule for a ARRAY type
array_type = wrap(BQArray, ('ARRAY', '<', bigquery_type, '>'))

# Grammar rule for a STRUCT type
struct_type = wrap(_create_struct_type,
                   ('STRUCT', '<',
                    separated_sequence([(identifier, bigquery_type),
                                        bigquery_type], ','), '>'))
