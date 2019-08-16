# Copyright 2019 Verily Life Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
