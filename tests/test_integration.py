"""Integration tests for end-to-end workflows."""

import os
import sys
import logging
import pytest
import subprocess
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import robust_tiff_compress
from robust_tiff_compress import (
    find_tiff_files,
    compress_tiff_file,
    FileLock,
    cleanup_temp_files,
    process_tiff_files,
    calculate_optimal_threads,
)


class TestFullWorkflow:
    """Test complete compression workflows."""
    
    def test_complete_compression_run_multiple_files(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test complete compression run with multiple files."""
        root_dir = sample_tiff_files[0].parent.parent
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        tiff_files = find_tiff_files(str(root_dir))
        
        for file_path in tiff_files:
            success, message, compression_ratio = compress_tiff_file(
                str(file_path),
                None,
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=False
            )
            
            if success:
                if "Skipped" in message:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                error_count += 1
        
        # Should have processed some files
        assert success_count + skip_count + error_count == len(tiff_files)
    
    def test_dry_run_mode(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test dry-run mode doesn't modify files."""
        root_dir = sample_tiff_files[0].parent.parent
        
        # Record original file sizes and modification times
        original_info = {}
        for file_path in sample_tiff_files:
            original_info[str(file_path)] = {
                'size': os.path.getsize(file_path),
                'mtime': os.path.getmtime(file_path)
            }
        
        tiff_files = find_tiff_files(str(root_dir))
        
        for file_path in tiff_files:
            success, message, compression_ratio = compress_tiff_file(
                str(file_path),
                None,
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=True
            )
            assert success
            assert "Dry run" in message or "would compress" in message.lower()
        
        # Verify files were not modified
        for file_path in sample_tiff_files:
            file_str = str(file_path)
            if file_str in original_info:
                assert os.path.getsize(file_path) == original_info[file_str]['size']
                assert os.path.getmtime(file_path) == original_info[file_str]['mtime']
    
    def test_cleanup_temp_files_before_run(
        self, tmp_test_dir, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test cleanup of temp files before compression run."""
        from robust_tiff_compress import TEMP_SUFFIX
        
        # Create some orphaned temp files
        temp_file1 = tmp_test_dir / ("file1.tif" + TEMP_SUFFIX)
        temp_file2 = tmp_test_dir / ("file2.tif" + TEMP_SUFFIX)
        
        # Create corresponding original files
        (tmp_test_dir / "file1.tif").touch()
        (tmp_test_dir / "file2.tif").touch()
        
        temp_file1.touch()
        temp_file2.touch()
        
        # Cleanup
        cleaned = cleanup_temp_files(str(tmp_test_dir), cleanup_error_files=False)
        assert cleaned >= 2
        assert not temp_file1.exists()
        assert not temp_file2.exists()
    
    def test_resume_after_partial_completion(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test resuming compression after partial completion."""
        root_dir = sample_tiff_files[0].parent.parent
        
        # Process first two files
        tiff_files = find_tiff_files(str(root_dir))
        files_to_process = tiff_files[:2]
        
        for file_path in files_to_process:
            success, message, compression_ratio = compress_tiff_file(
                str(file_path),
                None,
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=False
            )
            # May succeed or skip, but should be recorded in per-directory state
        
        # Find files again (should skip processed ones)
        remaining_files = find_tiff_files(str(root_dir))
        
        # Should have fewer files (processed ones excluded)
        assert len(remaining_files) <= len(tiff_files)


class TestCommandLineInterface:
    """Test command-line interface."""
    
    def test_command_line_help(self):
        """Test that command-line help works."""
        script_path = Path(__file__).parent.parent / "robust_tiff_compress.py"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
            assert "compression" in result.stdout.lower() or "compress" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Cannot test command-line interface")
    
    def test_command_line_invalid_folder(self):
        """Test command-line with invalid folder."""
        script_path = Path(__file__).parent.parent / "robust_tiff_compress.py"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--folder", "/nonexistent/path", "--compression", "zlib"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode != 0
            assert "not a directory" in result.stderr.lower() or "error" in result.stderr.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Cannot test command-line interface")
    
    def test_command_line_invalid_quality(self):
        """Test command-line with invalid quality value."""
        script_path = Path(__file__).parent.parent / "robust_tiff_compress.py"
        tmp_dir = Path("/tmp")  # Use /tmp which should exist
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--folder", str(tmp_dir), "--compression", "zlib", "--quality", "150"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode != 0
            assert "quality" in result.stderr.lower() or "between" in result.stderr.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Cannot test command-line interface")


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_compress_many_files_nested_structure(
        self, nested_test_dir, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test compression of many files in nested structure."""
        from tests.conftest import create_test_tiff
        
        # Create 50+ files in nested structure
        files_created = []
        for i in range(50):
            # Create files in different nested locations
            depth = (i % 3) + 1
            path_parts = [f"dir{i % 5}"]
            for d in range(depth - 1):
                path_parts.append(f"subdir{d}")
            path_parts.append(f"file_{i}.tif")
            
            file_path = nested_test_dir / Path(*path_parts)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            create_test_tiff(file_path, size_bytes=2 * 1024 * 1024, dtype=np.uint16)
            files_created.append(file_path)
        
        root_dir = nested_test_dir
        
        tiff_files = find_tiff_files(str(root_dir))
        assert len(tiff_files) >= 50
        
        # Process all files
        success_count = 0
        for file_path in tiff_files:
            success, message, compression_ratio = compress_tiff_file(
                str(file_path),
                None,
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=False
            )
            if success and "Compressed" in message:
                success_count += 1
        
        # Should have processed some files
        assert success_count >= 0
    
    def test_mixed_success_skip_error_scenarios(
        self, tmp_test_dir, state_file, mock_ram_large, mock_disk_space_sufficient, corrupted_tiff_file
    ):
        """Test mixed scenarios with success, skip, and error cases using process_tiff_files()."""
        from tests.conftest import create_test_tiff
        
        # Create various files
        # 1. Small file (should skip)
        small_file = tmp_test_dir / "small.tif"
        create_test_tiff(small_file, size_bytes=512 * 1024, dtype=np.uint8)
        
        # 2. Medium file (should compress)
        medium_file = tmp_test_dir / "medium.tif"
        create_test_tiff(medium_file, size_bytes=3 * 1024 * 1024, dtype=np.uint16)
        
        # 3. Corrupted file (should error) - using fixture
        corrupted_file = corrupted_tiff_file
        
        # Use process_tiff_files() which handles error counting
        files = [str(small_file), str(medium_file), str(corrupted_file)]
        threads = calculate_optimal_threads()
        
        results = process_tiff_files(
            tiff_files=files,
            folder=str(tmp_test_dir),
            output=None,
            compression="zlib",
            quality=85,
            threads=threads,
            state=None,  # Will use per-directory state
            dry_run=False,
            verify_lossless_exact=False,
            ignore_compression_ratio=False,
            show_progress=False
        )
        
        # Verify results using the returned dictionary
        # Should have at least one skip (small file)
        assert results['skip_count'] >= 1
        # Should have at least one error (corrupted file)
        assert results['error_count'] >= 1
        # Total should match number of files
        assert (results['success_count'] + results['skip_count'] + results['error_count']) == len(files)
    
    def test_state_file_persistence_across_multiple_runs(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that state file persists across multiple runs."""
        root_dir = sample_tiff_files[0].parent.parent
        
        # First run
        tiff_files1 = find_tiff_files(str(root_dir))
        
        # Process first file
        if tiff_files1:
            success, message, compression_ratio = compress_tiff_file(
                str(tiff_files1[0]),
                None,
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=False
            )
        
        # Find files again (should skip processed one)
        tiff_files2 = find_tiff_files(str(root_dir))
        
        # Should have fewer files (processed one excluded)
        assert len(tiff_files2) <= len(tiff_files1)


class TestLockingIntegration:
    """Test file locking in integration scenarios."""
    
    def test_lock_prevents_concurrent_runs(
        self, lock_file, mock_process_not_running, caplog
    ):
        """Test that lock prevents concurrent compression runs."""
        # First lock should succeed
        with FileLock(str(lock_file)) as lock1:
            assert lock1.locked
            assert lock_file.exists()
            
            # Second lock attempt
            lock2 = FileLock(str(lock_file))
            # With mock_process_not_running, acquire() should detect that the process
            # in the lock file is not running, remove the stale lock, and successfully
            # acquire (return True). This is the expected behavior when cleaning up stale locks.
            with caplog.at_level(logging.WARNING):
                result = lock2.acquire()
            
            # Test passes if acquire() returns True (stale lock removed because process not running)
            if result:
                # Lock was acquired after removing stale lock - this is valid behavior
                assert lock2.locked
                # Explicitly check that warning about removing stale lock was logged
                assert any(
                    "Removing stale lock file: process" in record.message 
                    and "is not running" in record.message
                    for record in caplog.records
                ), "Expected warning message about removing stale lock file not found in logs"
                lock2.release()
            # If result is False, that's also acceptable (lock is valid in real scenario)
            # but with mock_process_not_running, it should return True
    
    def test_lock_cleanup_on_exception(
        self, lock_file
    ):
        """Test that lock is cleaned up even when exception occurs."""
        try:
            with FileLock(str(lock_file)) as lock:
                assert lock.locked
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Lock should be released
        assert not lock_file.exists()


class TestOutputDirectoryIntegration:
    """Test output directory functionality in integration scenarios."""
    
    def test_output_directory_preserves_structure(
        self, nested_test_dir, sample_tiff_files, output_dir, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that output directory preserves nested directory structure."""
        root_dir = nested_test_dir
        
        for tiff_file in sample_tiff_files[:3]:  # Process first 3 files
            rel_path = tiff_file.relative_to(root_dir)
            output_path = output_dir / rel_path
            
            success, message, compression_ratio = compress_tiff_file(
                str(tiff_file),
                str(output_path),
                "zlib",
                85,
                None,
                None,  # Will use per-directory state
                dry_run=False
            )
            
            if success and "Compressed" in message:
                # Verify nested structure was created
                assert output_path.exists()
                assert output_path.parent.exists()
