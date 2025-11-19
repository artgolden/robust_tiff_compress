"""Tests for utility functions."""

import os
import pytest
import numpy as np
import tifffile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open

import robust_tiff_compress
from robust_tiff_compress import (
    get_available_ram,
    check_disk_space,
    is_process_running,
    verify_tiff_file,
    arrays_equal_exact,
    find_tiff_files,
    get_state_file_for_directory,
    CompressionState,
)


class TestRAMDetection:
    """Test RAM detection functions."""
    
    def test_get_available_ram_with_psutil(self, mock_ram_psutil):
        """Test get_available_ram() with psutil available."""
        # Mock psutil to be available
        with patch('robust_tiff_compress.PSUTIL_AVAILABLE', True):
            ram = get_available_ram()
            assert ram > 0
            assert ram == 8 * 1024 * 1024 * 1024  # 8GB from mock
    
    def test_get_available_ram_fallback_proc_meminfo(self, mock_ram_proc_meminfo):
        """Test get_available_ram() fallback to /proc/meminfo."""
        # psutil not available, should use /proc/meminfo
        ram = get_available_ram()
        assert ram > 0
        # Should parse from mock content: 8388608 kB = 8GB
        assert ram == 8 * 1024 * 1024 * 1024
    
    def test_get_available_ram_failure_handling(self):
        """Test get_available_ram() failure handling."""
        # Mock both psutil and /proc/meminfo to fail
        with patch('robust_tiff_compress.PSUTIL_AVAILABLE', False):
            with patch('builtins.open', side_effect=IOError("Cannot read /proc/meminfo")):
                with pytest.raises(RuntimeError, match="Failed to detect RAM"):
                    get_available_ram()
    
    def test_get_available_ram_meminfo_not_found(self):
        """Test get_available_ram() when MemTotal not found in /proc/meminfo."""
        mock_content = "MemFree: 1000000 kB\n"
        with patch('robust_tiff_compress.PSUTIL_AVAILABLE', False):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.__iter__.return_value = iter(mock_content.splitlines())
                mock_open.return_value = mock_file
                
                with pytest.raises(RuntimeError, match="MemTotal not found"):
                    get_available_ram()


class TestDiskSpaceChecking:
    """Test disk space checking functions."""
    
    def test_check_disk_space_with_psutil(self, mock_disk_space_psutil, tmp_test_dir):
        """Test check_disk_space() with psutil."""
        has_space, available = check_disk_space(str(tmp_test_dir), 10 * 1024 * 1024)
        assert has_space is True
        assert available > 0
    
    def test_check_disk_space_fallback_shutil(self, tmp_test_dir):
        """Test check_disk_space() fallback to shutil.disk_usage."""
        # Mock psutil to fail
        with patch('psutil.disk_usage', side_effect=Exception("psutil error")):
            has_space, available = check_disk_space(str(tmp_test_dir), 10 * 1024 * 1024)
            # Should use shutil fallback
            assert isinstance(has_space, bool)
            assert available >= 0
    
    def test_check_disk_space_fallback_statvfs(self, tmp_test_dir):
        """Test check_disk_space() fallback to statvfs."""
        # Mock both psutil and shutil to fail
        with patch('psutil.disk_usage', side_effect=Exception("psutil error")):
            with patch('shutil.disk_usage', side_effect=AttributeError("No disk_usage")):
                # Should use statvfs fallback
                has_space, available = check_disk_space(str(tmp_test_dir), 10 * 1024 * 1024)
                assert isinstance(has_space, bool)
                assert available >= 0
    
    def test_check_disk_space_all_failures(self, tmp_test_dir):
        """Test check_disk_space() when all methods fail."""
        # Mock all methods to fail
        with patch('psutil.disk_usage', side_effect=Exception("psutil error")):
            with patch('shutil.disk_usage', side_effect=AttributeError("No disk_usage")):
                with patch('os.statvfs', side_effect=OSError("statvfs failed")):
                    # Should assume sufficient space and return True
                    has_space, available = check_disk_space(str(tmp_test_dir), 10 * 1024 * 1024)
                    assert has_space is True
                    assert available == 0


class TestProcessChecking:
    """Test process checking functions."""
    
    def test_is_process_running_with_psutil(self, mock_process_psutil):
        """Test is_process_running() with psutil."""
        with patch('robust_tiff_compress.PSUTIL_AVAILABLE', True):
            result = is_process_running(12345)
            assert isinstance(result, bool)
    
    def test_is_process_running_fallback_os_kill(self, mock_process_os_kill):
        """Test is_process_running() fallback to os.kill."""
        # psutil not available, should use os.kill
        result = is_process_running(os.getpid())
        # os.kill with signal 0 doesn't kill, just checks
        # Should return True for current process
        assert isinstance(result, bool)
    
    def test_is_process_running_nonexistent_pid(self):
        """Test is_process_running() with nonexistent PID."""
        # Use a very large PID that likely doesn't exist
        nonexistent_pid = 99999999
        result = is_process_running(nonexistent_pid)
        assert result is False
    
    def test_is_process_running_current_pid(self):
        """Test is_process_running() with current process PID."""
        result = is_process_running(os.getpid())
        assert result is True


class TestFileVerification:
    """Test file verification functions."""
    
    def test_verify_tiff_file_valid(self, medium_tiff_file):
        """Test verify_tiff_file() with valid TIFF file."""
        success, array = verify_tiff_file(str(medium_tiff_file))
        assert success is True
        assert array is not None
        assert isinstance(array, np.ndarray)
        assert array.size > 0
    
    def test_verify_tiff_file_invalid(self, corrupted_tiff_file):
        """Test verify_tiff_file() with invalid TIFF file."""
        success, array = verify_tiff_file(str(corrupted_tiff_file))
        assert success is False
        assert array is None
    
    def test_verify_tiff_file_empty(self, empty_tiff_file):
        """Test verify_tiff_file() with empty file."""
        success, array = verify_tiff_file(str(empty_tiff_file))
        assert success is False
        assert array is None
    
    def test_verify_tiff_file_nonexistent(self, tmp_test_dir):
        """Test verify_tiff_file() with nonexistent file."""
        nonexistent = tmp_test_dir / "nonexistent.tif"
        success, array = verify_tiff_file(str(nonexistent))
        assert success is False
        assert array is None
    
    def test_verify_tiff_file_exception_handling(self, tmp_test_dir):
        """Test verify_tiff_file() exception handling."""
        # Create a file that will cause tifffile to raise exception
        bad_file = tmp_test_dir / "bad.tif"
        bad_file.write_bytes(b"invalid data")
        
        success, array = verify_tiff_file(str(bad_file))
        assert success is False
        assert array is None


class TestArrayComparison:
    """Test array comparison functions."""
    
    def test_arrays_equal_exact_matching(self):
        """Test arrays_equal_exact() with matching arrays."""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        arr2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        
        assert arrays_equal_exact(arr1, arr2) is True
    
    def test_arrays_equal_exact_different_values(self):
        """Test arrays_equal_exact() with different values."""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        arr2 = np.array([[1, 2, 3], [4, 5, 7]], dtype=np.uint16)
        
        assert arrays_equal_exact(arr1, arr2) is False
    
    def test_arrays_equal_exact_different_shapes(self):
        """Test arrays_equal_exact() with different shapes."""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        arr2 = np.array([[1, 2], [4, 5]], dtype=np.uint16)
        
        assert arrays_equal_exact(arr1, arr2) is False
    
    def test_arrays_equal_exact_different_dtypes(self):
        """Test arrays_equal_exact() with different dtypes."""
        arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        arr2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        
        assert arrays_equal_exact(arr1, arr2) is False
    
    def test_arrays_equal_exact_empty_arrays(self):
        """Test arrays_equal_exact() with empty arrays."""
        arr1 = np.array([], dtype=np.uint16)
        arr2 = np.array([], dtype=np.uint16)
        
        assert arrays_equal_exact(arr1, arr2) is True
    
    def test_arrays_equal_exact_exception_handling(self):
        """Test arrays_equal_exact() exception handling."""
        # Create arrays that will cause exception in comparison
        arr1 = np.array([[1, 2, 3]], dtype=np.uint16)
        arr2 = None  # Invalid input
        
        # Should handle exception gracefully
        result = arrays_equal_exact(arr1, arr2)
        assert result is False


class TestFileFinding:
    """Test file finding functions."""

    def test_find_tiff_files_nested_directories(self, nested_test_dir, sample_tiff_files):
        """Test find_tiff_files() with nested directory structure."""
        root_dir = nested_test_dir

        tiff_files = find_tiff_files(str(root_dir))

        # Should find all sample files
        found_paths = set(tiff_files)
        sample_paths = {str(f) for f in sample_tiff_files}

        assert found_paths == sample_paths

    def test_find_tiff_files_skip_processed(self, nested_test_dir, sample_tiff_files):
        """Test find_tiff_files() skipping already processed files."""
        state_file = get_state_file_for_directory(str(nested_test_dir))
        state = CompressionState(str(state_file))
        root_dir = nested_test_dir

        # Mark some files as processed
        state.mark_processed(str(sample_tiff_files[-1]), 2.0, "zlib", 1000000, 500000)
        state.mark_processed(str(sample_tiff_files[-2]), 2.5, "zlib", 2000000, 800000)

        tiff_files = find_tiff_files(str(root_dir))

        # Should not include processed files
        found_paths = set(tiff_files)
        processed_paths = {str(sample_tiff_files[-1]), str(sample_tiff_files[-2])}

        assert not (found_paths & processed_paths)
        assert len(tiff_files) == len(sample_tiff_files) - 2

    def test_find_tiff_files_skip_hidden_directories(self, tmp_test_dir):
        """Test find_tiff_files() skipping hidden directories."""
        from tests.conftest import create_test_tiff

        # Create files in hidden directory
        hidden_dir = tmp_test_dir / ".hidden"
        hidden_dir.mkdir()
        hidden_file = hidden_dir / "file.tif"
        create_test_tiff(hidden_file, size_bytes=2 * 1024 * 1024)

        # Create files in normal directory
        normal_dir = tmp_test_dir / "normal"
        normal_dir.mkdir()
        normal_file = normal_dir / "file.tif"
        create_test_tiff(normal_file, size_bytes=2 * 1024 * 1024)

        tiff_files = find_tiff_files(str(tmp_test_dir))

        # Should not find files in hidden directory
        found_paths = {str(f) for f in tiff_files}
        assert str(hidden_file) not in found_paths
        assert str(normal_file) in found_paths

    def test_find_tiff_files_case_insensitive_extensions(self, tmp_test_dir):
        """Test find_tiff_files() with case-insensitive extensions."""
        from tests.conftest import create_test_tiff

        # Create files with different case extensions
        file1 = tmp_test_dir / "file1.TIF"
        file2 = tmp_test_dir / "file2.TIFF"
        file3 = tmp_test_dir / "file3.tif"
        file4 = tmp_test_dir / "file4.tiff"

        for f in [file1, file2, file3, file4]:
            create_test_tiff(f, size_bytes=1 * 1024 * 1024)

        tiff_files = find_tiff_files(str(tmp_test_dir))

        # Should find all case variations
        found_paths = {str(f) for f in tiff_files}
        assert len(found_paths) == 4
        assert all(str(f) in found_paths for f in [file1, file2, file3, file4])

    def test_find_tiff_files_empty_directory(self, tmp_test_dir):
        """Test find_tiff_files() with empty directory."""
        tiff_files = find_tiff_files(str(tmp_test_dir))

        assert len(tiff_files) == 0

    def test_find_tiff_files_skip_state_file_directory(self, tmp_test_dir):
        """Test find_tiff_files() skipping directory named after state file."""
        from tests.conftest import create_test_tiff

        # Create a directory with state file name
        state_dir = tmp_test_dir / "_compression_state.json"
        state_dir.mkdir()
        file_in_state_dir = state_dir / "file.tif"
        create_test_tiff(file_in_state_dir, size_bytes=1 * 1024 * 1024)

        # Create file in normal directory
        normal_file = tmp_test_dir / "file.tif"
        create_test_tiff(normal_file, size_bytes=1 * 1024 * 1024)

        tiff_files = find_tiff_files(str(tmp_test_dir))

        # Should skip directory with state file name
        found_paths = {str(f) for f in tiff_files}
        assert str(file_in_state_dir) not in found_paths
        assert str(normal_file) in found_paths
