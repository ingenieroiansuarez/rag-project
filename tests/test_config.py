"""
Tests for configuration management.
"""
import os

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_load_with_valid_token():
    """Test that settings load correctly when HF_TOKEN is provided."""
    # Temporarily set environment variable for the test
    os.environ["HF_TOKEN"] = "hf_test_token123"

    # Initialize settings (should read from environment)
    settings = Settings()

    assert settings.hf_token == "hf_test_token123"
    assert settings.log_level == "INFO"  # Default should be intact

    # Cleanup
    del os.environ["HF_TOKEN"]


def test_settings_fail_without_token():
    """Test that settings raise a validation error when HF_TOKEN is missing."""
    # Ensure it's not set
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    with pytest.raises(ValidationError):
        Settings()
