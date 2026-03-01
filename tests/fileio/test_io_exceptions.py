"""Tests for IO exception hierarchy."""

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError


def test_io_format_error_message():
    """Test IOFormatError message formatting."""
    err = IOFormatError("Missing required group: obs")
    assert "Missing required group" in str(err)
    assert isinstance(err, Exception)


def test_io_write_error_message():
    """Test IOWriteError message formatting."""
    err = IOWriteError("Disk full", path="/data/output.h5")
    assert "Disk full" in str(err)
    assert "/data/output.h5" in str(err)


def test_io_password_error():
    """Test IOPasswordError for encrypted files."""
    err = IOPasswordError("File is password protected")
    assert "password" in str(err).lower()


def test_exception_inheritance():
    """Test all IO exceptions inherit from ScpTensorError."""
    from scptensor.core.exceptions import ScpTensorError

    assert issubclass(IOFormatError, ScpTensorError)
    assert issubclass(IOWriteError, ScpTensorError)
    assert issubclass(IOPasswordError, ScpTensorError)
