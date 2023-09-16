"""PyTest Configurations for Functional Testing

References:
- https://stackoverflow.com/questions/44377358/how-can-i-display-the-test-name-after-the-test-using-pytest
- https://docs.pytest.org/en/stable/fixture.html
"""

import pytest


@pytest.fixture(scope="session",)
def env_setup(monkeypatch):
    """Setup Environment Variables through PyTest's `monkeypatch` feature.

    Configure this for your repository. See an example below of how to use
        the `pytest` "`monkeypatch`" feature -- postfixes.
    """
    # monkeypatch.setenv(
    #     "GCLOUD_PROJECT",
    #     "samet-project",
    # )
    return None
