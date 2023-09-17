"""Distutils Setup Module for `tests_scripting` Directory

Note that this file is only to support local scripting, and
`poetry` should normally be used for actual building
and distribution of the `claim_analytics` package.
"""
from distutils.core import setup


setup(
    name="temporal_consistency",
    version="1.0.1",
    packages=["temporal_consistency"],
)
