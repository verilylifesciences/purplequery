def py2and3_test(**kwargs):
    original_name = kwargs.pop("name")
    kwargs["main"] = original_name + ".py"
    py2_name = original_name + "_py2"
    py3_name = original_name + "_py3"

    native.py_test(
        name = py2_name,
        python_version = "PY2",
        **kwargs
    )

    native.py_test(
        name = py3_name,
        python_version = "PY3",
        **kwargs
    )

    native.test_suite(
        name = original_name,
        tests = [py2_name, py3_name],
    )
