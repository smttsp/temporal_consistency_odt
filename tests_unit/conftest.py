"""PyTest Configurations for Unit-Testing

References:
- https://stackoverflow.com/questions/44377358/how-can-i-display-the-test-name-after-the-test-using-pytest
- https://docs.pytest.org/en/stable/fixture.html
"""

import pytest


@pytest.fixture(scope="session",)
def env_setup(monkeypatch):
    """Setup Environment Variables through PyTest's `monkeypatch` feature.

    """
    # monkeypatch.setenv(
    #     "GCLOUD_PROJECT",
    #     "samet-project",
    # )
    return None
