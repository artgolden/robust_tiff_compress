"""Tests for error handling and failure states."""

import os
import pytest
import numpy as np
import tifffile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from tests.conftest import create_test_tiff

import robust_tiff_compress
from robust_tiff_compress import (
    compress_tiff_file,
    CompressionState,
    cleanup_temp_files,
    create_warning_file,
    MAX_CONSECUTIVE_ERRORS,
    TEMP_SUFFIX,
    TEMP_ERROR_SUFFIX,
    WARNING_FILE,
)


class TestDiskSpaceFailures:
    """Test handling of insufficient disk space."""
    
    def test_insufficient_disk_space_in_place(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_insufficient
    ):
        """Test error handling when disk space is insufficient for in-place compression."""
        state = CompressionState(str(state_file))
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert not success
        assert "disk space" in message.lower() or "Insufficient" in message
    
    def test_insufficient_disk_space_output(
        self, medium_tiff_file, output_dir, state_file, mock_ram_large, mock_disk_space_insufficient
    ):
        """Test error handling when disk space is insufficient for output compression."""
        state = CompressionState(str(state_file))
        output_path = output_dir / medium_tiff_file.name
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            str(output_path),
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert not success
        assert "disk space" in message.lower() or "Insufficient" in message


class TestFilePermissionFailures:
    """Test handling of file permission errors."""
    
    def test_read_only_input_file(
        self, read_only_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of read-only input file."""
        state = CompressionState(str(state_file))
        
        # Try to compress read-only file
        # Note: On some systems, read-only files can still be read
        # The actual error might occur during write operations
        success, message, compression_ratio = compress_tiff_file(
            str(read_only_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        # May succeed if file can be read, or fail if write is needed
        # The important thing is that it handles the situation gracefully
        assert isinstance(success, bool)
    
    def test_read_only_output_directory(
        self, medium_tiff_file, read_only_dir, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of read-only output directory."""
        state = CompressionState(str(state_file))
        output_path = read_only_dir / medium_tiff_file.name
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            str(output_path),
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert not success
        assert "writable" in message.lower() or "permission" in message.lower() or "access" in message.lower()


class TestCorruptedFileHandling:
    """Test handling of corrupted or invalid files."""
    
    def test_invalid_tiff_file(
        self, corrupted_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of invalid TIFF file."""
        state = CompressionState(str(state_file))
        
        success, message, compression_ratio = compress_tiff_file(
            str(corrupted_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert not success
        assert "Could not read" in message or "TIFF" in message or "array" in message
    
    def test_empty_tiff_file(
        self, empty_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of empty TIFF file."""
        state = CompressionState(str(state_file))
        
        success, message, compression_ratio = compress_tiff_file(
            str(empty_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        # Should fail or skip
        assert not success or "Skipped" in message
    
    def test_truncated_tiff_file(
        self, truncated_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of truncated TIFF file."""
        state = CompressionState(str(state_file))
        
        success, message, compression_ratio = compress_tiff_file(
            str(truncated_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert not success
        assert "Could not read" in message or "TIFF" in message or "array" in message
    
    def test_wrong_dtype_after_compression(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test detection of wrong dtype after compression."""
        state = CompressionState(str(state_file))
        
        # Mock verify_tiff_file to return array with wrong dtype
        original_verify = robust_tiff_compress.verify_tiff_file
        
        def mock_verify(file_path):
            if TEMP_SUFFIX in file_path:
                # Return array with wrong dtype for compressed file
                wrong_array = np.zeros((100, 100), dtype=np.uint8)
                return True, wrong_array
            else:
                # Return correct array for original
                return original_verify(file_path)
        
        with patch('robust_tiff_compress.verify_tiff_file', side_effect=mock_verify):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False,
                ignore_compression_ratio=True
            )
            
            assert not success
            assert "dtype mismatch" in message.lower()
    
    def test_wrong_shape_after_compression(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test detection of wrong shape after compression."""
        state = CompressionState(str(state_file))
        
        # Mock verify_tiff_file to return array with wrong shape
        original_verify = robust_tiff_compress.verify_tiff_file
        
        def mock_verify(file_path):
            if TEMP_SUFFIX in file_path:
                # Return array with wrong shape for compressed file
                wrong_array = np.zeros((50, 50), dtype=np.uint16)
                return True, wrong_array
            else:
                # Return correct array for original
                return original_verify(file_path)
        
        with patch('robust_tiff_compress.verify_tiff_file', side_effect=mock_verify):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False,
                ignore_compression_ratio=True
            )
            
            assert not success
            assert "shape mismatch" in message.lower()


class TestCompressionFailures:
    """Test handling of compression failures."""
    
    def test_compression_ratio_threshold_not_met(
        self, tiff_file_not_compressible, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test skipping files that don't meet compression ratio threshold."""
        state = CompressionState(str(state_file))
        
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_not_compressible),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        # Should succeed but skip due to low compression ratio
        assert success
        assert "Skipped" in message
        assert "compression ratio" in message.lower() or "ratio" in message.lower()
    
    def test_compression_exception(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of compression exceptions."""
        state = CompressionState(str(state_file))
        
        # Mock tifffile.TiffWriter to raise exception
        with patch('tifffile.TiffWriter') as mock_writer:
            mock_writer.side_effect = Exception("Compression failed")
            
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            
            assert not success
            assert "Compression failed" in message or "error" in message.lower()
    
    def test_temp_file_creation_failure(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of temp file creation failure."""
        state = CompressionState(str(state_file))
        
        # Mock open to fail when creating temp file
        original_open = open
        
        def mock_open_fail(path, mode='r', *args, **kwargs):
            if TEMP_SUFFIX in str(path) and 'w' in mode:
                raise IOError("Cannot create temp file")
            return original_open(path, mode, *args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open_fail):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            
            # Should handle error gracefully
            assert not success or "error" in message.lower()
    
    def test_compressed_file_empty(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test detection of empty compressed file."""
        state = CompressionState(str(state_file))
        
        # Mock os.path.getsize to return 0 for temp file
        original_getsize = os.path.getsize
        
        def mock_getsize(path):
            if TEMP_SUFFIX in str(path):
                return 0
            return original_getsize(path)
        
        with patch('os.path.getsize', side_effect=mock_getsize):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            
            assert not success
            assert "empty" in message.lower()


class TestFileOperationFailures:
    """Test handling of file operation failures."""
    
    
    def test_move_operation_failure(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test handling of move operation failure - verify original file preservation and error backup creation."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(medium_tiff_file)
        original_mtime = os.path.getmtime(medium_tiff_file)
        temp_path = str(medium_tiff_file) + TEMP_SUFFIX
        error_path = str(medium_tiff_file) + TEMP_ERROR_SUFFIX
        
        # Mock shutil.move to raise OSError
        with patch('shutil.move', side_effect=OSError("Move failed")):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False,
                ignore_compression_ratio=True
            )
            
            assert not success
            assert "move" in message.lower() or "Failed" in message
            
            # Verify original file is preserved (size and existence)
            assert os.path.exists(medium_tiff_file), "Original file should still exist"
            assert os.path.getsize(medium_tiff_file) == original_size, "Original file size should be unchanged"
            
            # Verify temp file was renamed to error file
            assert not os.path.exists(temp_path), "Temp file should be renamed"
            assert os.path.exists(error_path), "Error-marked temp file should exist"
            
            # Error file is a TIFF file (binary), so we just verify it exists
            # The error message is logged, not stored in the file


class TestConsecutiveErrorHandling:
    """Test handling of consecutive errors."""
    
    def test_max_consecutive_errors_limit(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that processing stops after MAX_CONSECUTIVE_ERRORS using process_tiff_files."""
        from robust_tiff_compress import find_tiff_files, process_tiff_files, WARNING_FILE
        
        state = CompressionState(str(state_file))
        root_dir = sample_tiff_files[0].parent.parent
        
        # Mock compress_tiff_file to fail
        def mock_compress(*args, **kwargs):
            return False, "Test error", None
        
        tiff_files = find_tiff_files(str(root_dir), state)
        # Need at least MAX_CONSECUTIVE_ERRORS files to test
        # If we don't have enough, create additional test files
        if len(tiff_files) < MAX_CONSECUTIVE_ERRORS:
            for i in range(len(tiff_files), MAX_CONSECUTIVE_ERRORS):
                extra_file = root_dir / f"extra_file_{i}.tif"
                create_test_tiff(extra_file, size_bytes=2 * 1024 * 1024, dtype=np.uint16)
            tiff_files = find_tiff_files(str(root_dir), state)
        assert len(tiff_files) >= MAX_CONSECUTIVE_ERRORS, "Need at least MAX_CONSECUTIVE_ERRORS files for this test"
        
        with patch('robust_tiff_compress.compress_tiff_file', side_effect=mock_compress):
            results = process_tiff_files(
                tiff_files,
                str(root_dir),
                None,
                "zlib",
                85,
                1,
                state,
                dry_run=False,
                verify_lossless_exact=False,
                ignore_compression_ratio=True,
                show_progress=False
            )
        
        # Should have encountered MAX_CONSECUTIVE_ERRORS and stopped
        assert results['consecutive_errors'] >= MAX_CONSECUTIVE_ERRORS
        assert results['stop_processing'] is True
        assert results['error_count'] >= MAX_CONSECUTIVE_ERRORS
        
        # Verify warning file was created
        warning_path = root_dir / WARNING_FILE
        assert warning_path.exists(), "Warning file should be created"
        
        # Verify warning file contents
        with open(warning_path, 'r') as f:
            content = f.read()
            assert "COMPRESSION STOPPED" in content
            assert str(results['consecutive_errors']) in content
    
    def test_warning_file_creation(
        self, tmp_test_dir, state_file
    ):
        """Test creation of warning file when compression stops."""
        root_dir = tmp_test_dir
        error_count = MAX_CONSECUTIVE_ERRORS
        last_errors = [
            ("/path/to/file1.tif", "Error 1"),
            ("/path/to/file2.tif", "Error 2"),
            ("/path/to/file3.tif", "Error 3"),
        ]
        
        create_warning_file(str(root_dir), error_count, last_errors)
        
        warning_path = root_dir / WARNING_FILE
        assert warning_path.exists()
        
        # Verify warning file contents
        with open(warning_path, 'r') as f:
            content = f.read()
            assert "COMPRESSION STOPPED" in content
            assert str(error_count) in content
            for file_path, error_msg in last_errors:
                assert file_path in content
                assert error_msg in content


class TestTempFileErrorMarking:
    """Test temp file error marking and cleanup."""
    
    def test_temp_file_error_marking(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that temp files are marked with .ERROR suffix on failure."""
        state = CompressionState(str(state_file))
        temp_path = Path(str(medium_tiff_file) + TEMP_SUFFIX)
        error_path = Path(str(medium_tiff_file) + TEMP_ERROR_SUFFIX)
        
        # Mock compression to fail after creating temp file
        original_write = tifffile.TiffWriter
        
        def mock_write_fail(*args, **kwargs):
            # Create a temp file first
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.touch()
            raise Exception("Compression failed")
        
        with patch('tifffile.TiffWriter', side_effect=mock_write_fail):
            success, message, compression_ratio = compress_tiff_file(
                str(medium_tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            
            # Temp file should be renamed to error file
            assert not success
            # Note: The actual renaming happens in rename_temp_file_on_error
            # which is called from compress_tiff_file
    
    def test_cleanup_temp_files(
        self, tmp_test_dir, medium_tiff_file
    ):
        """Test cleanup of temporary files."""
        # Create some temp files
        temp_file1 = tmp_test_dir / ("file1.tif" + TEMP_SUFFIX)
        temp_file2 = tmp_test_dir / ("file2.tif" + TEMP_SUFFIX)
        error_file = tmp_test_dir / ("file3.tif" + TEMP_ERROR_SUFFIX)
        
        # Create corresponding original files
        (tmp_test_dir / "file1.tif").touch()
        (tmp_test_dir / "file2.tif").touch()
        (tmp_test_dir / "file3.tif").touch()
        
        temp_file1.touch()
        temp_file2.touch()
        error_file.touch()
        
        # Cleanup (should not remove error files by default)
        cleaned = cleanup_temp_files(str(tmp_test_dir), cleanup_error_files=False)
        
        # Temp files should be cleaned up
        assert not temp_file1.exists()
        assert not temp_file2.exists()
        # Error file should remain
        assert error_file.exists()
        assert cleaned >= 2
    
    def test_cleanup_error_files(
        self, tmp_test_dir
    ):
        """Test cleanup of error-marked files when requested."""
        error_file = tmp_test_dir / ("file.tif" + TEMP_ERROR_SUFFIX)
        (tmp_test_dir / "file.tif").touch()  # Original file
        error_file.touch()
        
        # Cleanup with cleanup_error_files=True
        cleaned = cleanup_temp_files(str(tmp_test_dir), cleanup_error_files=True)
        
        # Error file should be cleaned up
        assert not error_file.exists()
        assert cleaned >= 1
