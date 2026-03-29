"""Basic import tests for llm_compress package."""

import pytest
from llm_compress import __version__


def test_version():
    """Test that version is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_import_package():
    """Test that the package can be imported."""
    import llm_compress
    assert hasattr(llm_compress, "__version__")
    assert hasattr(llm_compress, "quantize_model")
    assert hasattr(llm_compress, "get_backend")


def test_import_submodules():
    """Test that all submodules can be imported."""
    from llm_compress import quantization
    from llm_compress import backends
    from llm_compress import server
    
    assert hasattr(quantization, "quantize_model")
    assert hasattr(backends, "get_backend")
    assert hasattr(server, "create_app")


def test_cli_import():
    """Test that CLI module can be imported."""
    from llm_compress import cli
    assert hasattr(cli, "main")
