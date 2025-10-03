"""
Shared fixtures for integration tests.

Imports fixtures from e2e tests for consistency.
"""

import pytest

# Import all fixtures from e2e conftest
pytest_plugins = ["tests.e2e.conftest"]
