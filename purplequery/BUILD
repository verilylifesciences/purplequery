package(default_visibility = ["//visibility:public"])

load(":pytest.bzl", "py2and3_test")

py_library(
    name = "client",
    srcs = ["client.py"],
    deps = [
        ":bq_types",
        ":query",
    ],
)

py2and3_test(
    name = "client_test",
    srcs = ["client_test.py"],
    deps = [
        ":bq_types",
        ":client",
    ],
)

py_test(
    name = "bq_client_test",
    srcs = ["bq_client_test.py"],
    deps = [":client"],
)

py_library(
    name = "bq_abstract_syntax_tree",
    srcs = ["bq_abstract_syntax_tree.py"],
    deps = [
        ":bq_types",
    ],
)

py2and3_test(
    name = "bq_abstract_syntax_tree_test",
    srcs = ["bq_abstract_syntax_tree_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":dataframe_node",
        ":evaluatable_node",
    ],
)

py_library(
    name = "join",
    srcs = ["join.py"],
    deps = [
        ":binary_expression",
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":evaluatable_node",
    ],
)

py2and3_test(
    name = "join_test",
    srcs = ["join_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":dataframe_node",
        ":grammar",
        ":join",
        ":query_helper",
        ":tokenizer",
    ],
)

py_library(
    name = "evaluatable_node",
    srcs = ["evaluatable_node.py"],
    deps = [
        ":binary_expression",
        ":bq_abstract_syntax_tree",
        ":bq_types",
    ],
)

py2and3_test(
    name = "evaluatable_node_test",
    srcs = ["evaluatable_node_test.py"],
    deps = [
        ":binary_expression",
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":dataframe_node",
        ":evaluatable_node",
        ":grammar",
        ":join",
        ":tokenizer",
    ],
)

py_library(
    name = "dataframe_node",
    srcs = ["dataframe_node.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":join",
    ],
)

py2and3_test(
    name = "dataframe_node_test",
    srcs = ["dataframe_node_test.py"],
    deps = [
        ":binary_expression",
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":dataframe_node",
        ":evaluatable_node",
        ":grammar",
        ":join",
        ":tokenizer",
    ],
)

py_library(
    name = "bq_binary_operators",
    srcs = ["bq_binary_operators.py"],
    deps = [
        ":bq_types",
    ],
)

py_library(
    name = "binary_expression",
    srcs = ["binary_expression.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_binary_operators",
        ":bq_types",
    ],
)

py_library(
    name = "bq_operator",
    srcs = ["bq_operator.py"],
    deps = [
        ":binary_expression",
        ":bq_abstract_syntax_tree",
        ":bq_binary_operators",
        ":query_helper",
    ],
)

py2and3_test(
    name = "bq_operator_test",
    srcs = ["bq_operator_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_operator",
        ":bq_types",
        ":evaluatable_node",
        ":terminals",
    ],
)

py_library(
    name = "bq_types",
    srcs = ["bq_types.py"],
)

py2and3_test(
    name = "bq_types_test",
    srcs = ["bq_types_test.py"],
    deps = [":bq_types"],
)

py_library(
    name = "grammar",
    srcs = ["grammar.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_operator",
        ":dataframe_node",
        ":evaluatable_node",
        ":join",
        ":query_helper",
        ":terminals",
        ":type_grammar",
    ],
)

py2and3_test(
    name = "grammar_test",
    srcs = ["grammar_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":dataframe_node",
        ":evaluatable_node",
        ":grammar",
    ],
)

py_library(
    name = "query",
    srcs = ["query.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":grammar",
        ":query_helper",
        ":tokenizer",
    ],
)

py2and3_test(
    name = "query_test",
    srcs = ["query_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":grammar",
        ":query",
    ],
)

py_library(
    name = "query_helper",
    srcs = ["query_helper.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":terminals",
    ],
)

py2and3_test(
    name = "query_helper_test",
    srcs = ["query_helper_test.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":evaluatable_node",
        ":query_helper",
        ":terminals",
    ],
)

py_library(
    name = "patterns",
    srcs = ["patterns.py"],
)

py_library(
    name = "terminals",
    srcs = ["terminals.py"],
    deps = [
        ":bq_types",
        ":evaluatable_node",
        ":patterns",
    ],
)

py2and3_test(
    name = "terminals_test",
    srcs = ["terminals_test.py"],
    deps = [
        ":bq_types",
        ":evaluatable_node",
        ":terminals",
    ],
)

py_library(
    name = "tokenizer",
    srcs = ["tokenizer.py"],
    deps = [
        ":bq_operator",
        ":patterns",
    ],
)

py2and3_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.py"],
    deps = [":tokenizer"],
)

py_library(
    name = "type_grammar",
    srcs = ["type_grammar.py"],
    deps = [
        ":bq_abstract_syntax_tree",
        ":bq_types",
        ":query_helper",
        ":terminals",
    ],
)

py2and3_test(
    name = "type_grammar_test",
    srcs = ["type_grammar_test.py"],
    deps = [
        ":tokenizer",
        ":type_grammar",
    ],
)
