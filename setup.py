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

"""Package configuration."""

import platform
from setuptools import find_packages
from setuptools import setup

ANALYSIS_PY_UTILS = 'analysis-py-utils @ ' \
    'https://github.com/verilylifesciences/analysis-py-utils/archive/' \
    '11b06535c4d3973670d5b23ae3846f3601f00a1a.zip#egg=analysis-py-utils-1.0'
REQUIRED_PACKAGES = ['numpy==1.13.3',
                     'pandas==0.20.3',
                     ANALYSIS_PY_UTILS,
                     'google-api-core==1.6.0',
                     'google-cloud-bigquery==1.8.0',
                     'ddt>=1.2.1',
                     'six>=1.12.0',
                     'typing>=3.7.4']

setup(
    name='purplequery',
    version='1.0',
    license='Apache 2.0',
    author='Verily Life Sciences',
    url='https://github.com/verilylifesciences/purplequery',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Fake implementation of BigQuery using Pandas',
    scripts=[],
    requires=[])
