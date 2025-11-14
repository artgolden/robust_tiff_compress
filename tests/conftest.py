"""Shared fixtures and test utilities for TIFF compression tests."""

import os
import sys
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import tifffile

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robust_tiff_compress


# Constants from the module
MIN_FILE_SIZE = robust_tiff_compress.MIN_FILE_SIZE
COMPRESSION_RATIO_THRESHOLD = robust_tiff_compress.COMPRESSION_RATIO_THRESHOLD
RAM_SIZE_LIMIT_RATIO = robust_tiff_compress.RAM_SIZE_LIMIT_RATIO
STATE_FILE = robust_tiff_compress.STATE_FILE
TEMP_SUFFIX = robust_tiff_compress.TEMP_SUFFIX
TEMP_ERROR_SUFFIX = robust_tiff_compress.TEMP_ERROR_SUFFIX
LOCK_FILE = robust_tiff_compress.LOCK_FILE
WARNING_FILE = robust_tiff_compress.WARNING_FILE


@pytest.fixture
def tmp_test_dir(tmp_path):
    """Create a temporary test directory that's cleaned up after test."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def nested_test_dir(tmp_test_dir):
    """Create a nested directory structure for testing."""
    # Create nested structure: dir1/subdir1, dir1/subdir2/nested, dir2/subdir3
    dirs = [
        tmp_test_dir / "dir1" / "subdir1",
        tmp_test_dir / "dir1" / "subdir2" / "nested",
        tmp_test_dir / "dir2" / "subdir3",
        tmp_test_dir / "dir3",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return tmp_test_dir


def create_test_tiff(
    file_path: Path,
    size_bytes: Optional[int] = None,
    dtype: np.dtype = np.uint16,
    shape: Optional[Tuple[int, ...]] = None,
    pattern: str = "random",
    compression: Optional[str] = None
) -> Path:
    """
    Create a test TIFF file with specified characteristics.
    
    Args:
        file_path: Path where TIFF file should be created
        size_bytes: Target file size in bytes (approximate). If None, uses shape and dtype.
        dtype: NumPy dtype for the array (default: uint16)
        shape: Shape of the array. If None and size_bytes is set, calculates shape.
        pattern: Data pattern - "random", "zeros", "ones", "gradient", "checkerboard"
        compression: Compression to use when writing (None for uncompressed)
    
    Returns:
        Path to created file
    """
    # Calculate shape if not provided
    if shape is None:
        if size_bytes is None:
            # Default: 512x512 uint16 image
            shape = (512, 512)
        else:
            # Calculate shape to approximate target size
            bytes_per_pixel = np.dtype(dtype).itemsize
            num_pixels = size_bytes // bytes_per_pixel
            # Create roughly square image
            side = int(np.sqrt(num_pixels))
            shape = (side, side)
    
    # Generate array data based on pattern
    if pattern == "random":
        if dtype in (np.uint8, np.uint16):
            array = np.random.randint(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
        else:
            array = np.random.rand(*shape).astype(dtype)
    elif pattern == "zeros":
        array = np.zeros(shape, dtype=dtype)
    elif pattern == "ones":
        array = np.ones(shape, dtype=dtype)
    elif pattern == "gradient":
        array = np.linspace(0, np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0,
                          num=np.prod(shape), dtype=dtype).reshape(shape)
    elif pattern == "checkerboard":
        array = np.zeros(shape, dtype=dtype)
        checker = np.indices(shape).sum(axis=0) % 2
        max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        array[checker == 0] = max_val
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Write TIFF file
    with tifffile.TiffWriter(str(file_path)) as tiff:
        if compression:
            tiff.write(array, compression=compression)
        else:
            tiff.write(array)
    
    return file_path


@pytest.fixture
def small_tiff_file(tmp_test_dir):
    """Create a small TIFF file (< MIN_FILE_SIZE)."""
    file_path = tmp_test_dir / "small.tif"
    # Create file smaller than MIN_FILE_SIZE (1MB)
    create_test_tiff(file_path, size_bytes=512 * 1024, dtype=np.uint8)
    return file_path


@pytest.fixture
def medium_tiff_file(tmp_test_dir):
    """Create a medium-sized TIFF file (2-5MB)."""
    file_path = tmp_test_dir / "medium.tif"
    create_test_tiff(file_path, size_bytes=3 * 1024 * 1024, dtype=np.uint16)
    return file_path


@pytest.fixture
def large_tiff_file(tmp_test_dir):
    """Create a large TIFF file (10-20MB)."""
    file_path = tmp_test_dir / "large.tif"
    create_test_tiff(file_path, size_bytes=15 * 1024 * 1024, dtype=np.uint16)
    return file_path


@pytest.fixture
def very_large_tiff_file(tmp_test_dir):
    """Create a very large TIFF file (>20% of RAM, should be skipped)."""
    file_path = tmp_test_dir / "very_large.tif"
    # Create file that's larger than 20% of typical RAM (assuming 8GB RAM, 20% = 1.6GB)
    # For testing, we'll create a smaller file but mock RAM to be small
    create_test_tiff(file_path, size_bytes=100 * 1024 * 1024, dtype=np.uint16)
    return file_path


@pytest.fixture
def tiff_file_uint8(tmp_test_dir):
    """Create a uint8 TIFF file."""
    file_path = tmp_test_dir / "uint8.tif"
    create_test_tiff(file_path, dtype=np.uint8, shape=(1000, 1000), pattern="random")
    return file_path


@pytest.fixture
def tiff_file_uint16(tmp_test_dir):
    """Create a uint16 TIFF file."""
    file_path = tmp_test_dir / "uint16.tif"
    create_test_tiff(file_path, dtype=np.uint16, shape=(1000, 1000), pattern="random")
    return file_path


@pytest.fixture
def tiff_file_float32(tmp_test_dir):
    """Create a float32 TIFF file."""
    file_path = tmp_test_dir / "float32.tif"
    create_test_tiff(file_path, dtype=np.float32, shape=(1000, 1000), pattern="random")
    return file_path


@pytest.fixture
def tiff_file_3d(tmp_test_dir):
    """Create a 3D TIFF file (stack)."""
    file_path = tmp_test_dir / "3d_stack.tif"
    create_test_tiff(file_path, dtype=np.uint16, shape=(50, 512, 512), pattern="random")
    return file_path


@pytest.fixture
def tiff_file_compressible(tmp_test_dir):
    """Create a TIFF file that compresses well (repetitive pattern)."""
    file_path = tmp_test_dir / "compressible.tif"
    create_test_tiff(file_path, dtype=np.uint16, shape=(2000, 2000), pattern="checkerboard")
    return file_path


@pytest.fixture
def tiff_file_not_compressible(tmp_test_dir):
    """Create a TIFF file that doesn't compress well (random data)."""
    file_path = tmp_test_dir / "not_compressible.tif"
    create_test_tiff(file_path, dtype=np.uint16, shape=(1000, 1000), pattern="random")
    return file_path


@pytest.fixture
def state_file(tmp_test_dir):
    """Create a clean state file path."""
    return tmp_test_dir / STATE_FILE


@pytest.fixture
def existing_state_file(state_file):
    """Create a state file with some existing processed files."""
    state_data = {
        "processed": {
            "/path/to/file1.tif": {
                "compression_ratio": 2.5,
                "compression_type": "zlib",
                "original_size": 1000000,
                "compressed_size": 400000,
                "timestamp": "2024-01-01T00:00:00"
            },
            "/path/to/file2.tif": {
                "compression_ratio": 3.0,
                "compression_type": "jpeg_2000_lossy",
                "original_size": 2000000,
                "compressed_size": 666666,
                "timestamp": "2024-01-01T01:00:00"
            }
        }
    }
    with open(state_file, 'w') as f:
        json.dump(state_data, f)
    return state_file


@pytest.fixture
def corrupted_state_file(state_file):
    """Create a corrupted state file (invalid JSON)."""
    with open(state_file, 'w') as f:
        f.write("{ invalid json }")
    return state_file


@pytest.fixture
def mock_ram_large():
    """Mock get_available_ram() to return large RAM (16GB)."""
    with patch('robust_tiff_compress.get_available_ram', return_value=16 * 1024 * 1024 * 1024):
        yield


@pytest.fixture
def mock_ram_small():
    """Mock get_available_ram() to return small RAM (1GB) for testing size limits."""
    with patch('robust_tiff_compress.get_available_ram', return_value=1024 * 1024 * 1024):
        yield


@pytest.fixture
def mock_ram_psutil():
    """Mock psutil for RAM detection."""
    mock_virtual_memory = Mock()
    mock_virtual_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
    with patch('psutil.virtual_memory', return_value=mock_virtual_memory):
        yield


@pytest.fixture
def mock_ram_proc_meminfo():
    """Mock /proc/meminfo for RAM detection fallback."""
    mock_content = "MemTotal:        8388608 kB\n"
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = iter(mock_content.splitlines())
        mock_open.return_value = mock_file
        with patch('psutil.virtual_memory', side_effect=Exception("psutil not available")):
            with patch('robust_tiff_compress.PSUTIL_AVAILABLE', False):
                yield


@pytest.fixture
def mock_disk_space_sufficient():
    """Mock check_disk_space() to return sufficient space."""
    def _mock_check_disk_space(path, required_bytes):
        return True, required_bytes * 2
    
    with patch('robust_tiff_compress.check_disk_space', side_effect=_mock_check_disk_space):
        yield


@pytest.fixture
def mock_disk_space_insufficient():
    """Mock check_disk_space() to return insufficient space."""
    def _mock_check_disk_space(path, required_bytes):
        return False, required_bytes // 2
    
    with patch('robust_tiff_compress.check_disk_space', side_effect=_mock_check_disk_space):
        yield


@pytest.fixture
def mock_disk_space_psutil():
    """Mock psutil for disk space checking."""
    mock_disk_usage = Mock()
    mock_disk_usage.free = 100 * 1024 * 1024 * 1024  # 100GB free
    with patch('psutil.disk_usage', return_value=mock_disk_usage):
        yield


@pytest.fixture
def mock_process_running():
    """Mock is_process_running() to return True for a specific PID."""
    def _mock_is_running(pid):
        return pid == 12345
    
    with patch('robust_tiff_compress.is_process_running', side_effect=_mock_is_running):
        yield


@pytest.fixture
def mock_process_not_running():
    """Mock is_process_running() to return False for all PIDs."""
    with patch('robust_tiff_compress.is_process_running', return_value=False):
        yield


@pytest.fixture
def mock_process_psutil():
    """Mock psutil for process checking."""
    with patch('psutil.pid_exists', return_value=True):
        yield


@pytest.fixture
def mock_process_os_kill():
    """Mock os.kill for process checking fallback."""
    with patch('os.kill', return_value=None):
        with patch('robust_tiff_compress.PSUTIL_AVAILABLE', False):
            yield


@pytest.fixture
def corrupted_tiff_file(tmp_test_dir):
    """Create a corrupted TIFF file (invalid data)."""
    file_path = tmp_test_dir / "corrupted.tif"
    # Write invalid TIFF data
    with open(file_path, 'wb') as f:
        f.write(b"NOT A VALID TIFF FILE" + b"\x00" * 1000)
    return file_path


@pytest.fixture
def empty_tiff_file(tmp_test_dir):
    """Create an empty TIFF file."""
    file_path = tmp_test_dir / "empty.tif"
    file_path.touch()
    return file_path


@pytest.fixture
def truncated_tiff_file(tmp_test_dir, medium_tiff_file):
    """Create a truncated TIFF file (valid start, but cut off)."""
    truncated_path = tmp_test_dir / "truncated.tif"
    # Copy first half of a valid file
    with open(medium_tiff_file, 'rb') as src:
        data = src.read()
        with open(truncated_path, 'wb') as dst:
            dst.write(data[:len(data) // 2])
    return truncated_path


@pytest.fixture
def lock_file(tmp_test_dir):
    """Create a lock file path."""
    return tmp_test_dir / LOCK_FILE


@pytest.fixture
def existing_lock_file(lock_file):
    """Create an existing lock file."""
    with open(lock_file, 'w') as f:
        f.write(f"{os.getpid()}\n2024-01-01T00:00:00\n")
    return lock_file


@pytest.fixture
def stale_lock_file(lock_file):
    """Create a stale lock file (old timestamp)."""
    with open(lock_file, 'w') as f:
        f.write("99999\n2020-01-01T00:00:00\n")
    # Set file modification time to be old
    old_time = os.path.getmtime(lock_file) - (25 * 3600)  # 25 hours ago
    os.utime(lock_file, (old_time, old_time))
    return lock_file


@pytest.fixture
def output_dir(tmp_test_dir):
    """Create an output directory for compression tests."""
    out_dir = tmp_test_dir / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def read_only_dir(tmp_test_dir):
    """Create a read-only directory for permission testing."""
    ro_dir = tmp_test_dir / "readonly"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o555)  # Read and execute only
    yield ro_dir
    # Restore permissions for cleanup
    os.chmod(ro_dir, 0o755)


@pytest.fixture
def read_only_file(tmp_test_dir, medium_tiff_file):
    """Create a read-only file for permission testing."""
    ro_file = tmp_test_dir / "readonly.tif"
    shutil.copy(medium_tiff_file, ro_file)
    os.chmod(ro_file, 0o444)  # Read only
    yield ro_file
    # Restore permissions for cleanup
    os.chmod(ro_file, 0o644)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    # Save original handlers
    original_handlers = logging.root.handlers[:]
    yield
    # Restore original handlers
    logging.root.handlers = original_handlers


@pytest.fixture
def sample_tiff_files(nested_test_dir):
    """Create multiple TIFF files in nested directory structure."""
    files = []
    # Create files in different nested locations
    locations = [
        ("dir1", "subdir1", "file1.tif"),
        ("dir1", "subdir2", "nested", "file2.tif"),
        ("dir2", "subdir3", "file3.tif"),
        ("dir3", "file4.tif"),
        ("file5.tif"),  # Root level
    ]
    
    for location in locations:
        if isinstance(location, tuple):
            file_path = nested_test_dir / Path(*location)
        else:
            file_path = nested_test_dir / location
        file_path.parent.mkdir(parents=True, exist_ok=True)
        create_test_tiff(file_path, size_bytes=2 * 1024 * 1024, dtype=np.uint16)
        files.append(file_path)
    
    return files

