# config.py

# Directories or paths related to tests
TEST_DIRS = ["tests_unit"]

# Coverage settings
COV_PATHS = ["temporal_consistency"]
COV_OPTIONS = {
    "branch": True,
    "report": "term-missing",
    "fail_under": 5,
}

# Add any custom fixtures or settings as needed for your pytest setup
