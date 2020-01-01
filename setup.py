# Copyright 2019 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Package configuration."""

from setuptools import find_packages, setup

REQUIRED_PACKAGES = ['numpy==1.14.3',
                     'pandas==0.23.3',
                     'google-api-core==1.6.0',
                     'google-cloud-bigquery==1.8.0',
                     'ddt>=1.2.1',
                     'six>=1.12.0',
                     'typing>=3.7.4']

setup(
    name='purplequery',
    version='1.0',
    license='BSD',
    author='Verily Life Sciences',
    url='https://github.com/verilylifesciences/purplequery',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Fake implementation of BigQuery using Pandas',
    scripts=[],
    requires=[])
