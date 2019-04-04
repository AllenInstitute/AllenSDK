#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_internal
----------------------------------

Tests for `internal` module.
"""
import pytest


@pytest.fixture
def decorated_example():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """

def test_example(decorated_example):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    import allensdk.internal




