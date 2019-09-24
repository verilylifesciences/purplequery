# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest

from verily.bigquery_wrapper import bq_shared_tests

from client import Client as FakeClient


class BQClientTest(bq_shared_tests.BQSharedTests):
    """Run the real/mock shared tests on the fake BQ library."""

    @classmethod
    def setUpClass(cls, use_mocks=False):
        super(BQClientTest, cls).setUpClass(use_mocks=use_mocks,
                                            alternate_bq_client_class=FakeClient)


if __name__ == '__main__':
    unittest.main()
