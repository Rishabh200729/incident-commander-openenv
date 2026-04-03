"""
Shared pytest fixtures and configuration.
"""

import pytest
from server.environment import IncidentCommanderEnvironment


@pytest.fixture
def env():
    """Fresh environment instance."""
    e = IncidentCommanderEnvironment()
    yield e
    e.close()


@pytest.fixture
def easy_env():
    """Environment pre-reset to the easy task."""
    e = IncidentCommanderEnvironment()
    e.reset(task_name="single_service_failure", episode_id="test-easy")
    yield e
    e.close()


@pytest.fixture
def medium_env():
    """Environment pre-reset to the medium task."""
    e = IncidentCommanderEnvironment()
    e.reset(task_name="cascading_failure", episode_id="test-medium")
    yield e
    e.close()


@pytest.fixture
def hard_env():
    """Environment pre-reset to the hard task."""
    e = IncidentCommanderEnvironment()
    e.reset(task_name="hidden_root_cause", episode_id="test-hard")
    yield e
    e.close()
