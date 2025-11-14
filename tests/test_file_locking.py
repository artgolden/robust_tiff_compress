"""Tests for file locking and concurrent run prevention."""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

import robust_tiff_compress
from robust_tiff_compress import FileLock, LOCK_FILE


class TestLockAcquisition:
    """Test lock file acquisition."""
    
    def test_successful_lock_acquisition(self, lock_file):
        """Test successfully acquiring a lock when no lock exists."""
        lock = FileLock(str(lock_file))
        
        assert lock.acquire()
        assert lock.locked
        assert lock_file.exists()
        
        # Verify lock file contents
        with open(lock_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 2
            assert lines[0].strip().isdigit()  # PID
            # Second line should be timestamp
    
    def test_lock_file_creation_with_pid_and_timestamp(self, lock_file):
        """Test that lock file contains PID and timestamp."""
        lock = FileLock(str(lock_file))
        lock.acquire()
        
        with open(lock_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
            # First line should be PID
            assert lines[0].strip().isdigit()
            pid = int(lines[0].strip())
            assert pid == os.getpid()
            
            # Second line should be timestamp (ISO format)
            assert len(lines) >= 2
    
    def test_lock_detection_when_process_running(self, existing_lock_file, mock_process_running):
        """Test that lock is detected when another process is running."""
        # Mock the PID in the lock file to match the mocked running process
        with open(existing_lock_file, 'w') as f:
            f.write("12345\n2024-01-01T00:00:00\n")
        
        lock = FileLock(str(existing_lock_file))
        
        # Should not acquire lock because process 12345 is running
        assert not lock.acquire()
        assert not lock.locked
    
    def test_stale_lock_detection(self, stale_lock_file, mock_process_not_running):
        """Test detection and cleanup of stale lock files."""
        lock = FileLock(str(stale_lock_file))
        
        # Should acquire lock after removing stale lock
        assert lock.acquire()
        assert lock.locked
    
    def test_stale_lock_cleanup(self, stale_lock_file, mock_process_not_running):
        """Test that stale lock files are removed."""
        assert stale_lock_file.exists()
        
        lock = FileLock(str(stale_lock_file))
        lock.acquire()
        
        # Stale lock should be removed and new lock created
        assert stale_lock_file.exists()
        # Verify it's a new lock (different PID or timestamp)
        with open(stale_lock_file, 'r') as f:
            pid = int(f.readline().strip())
            assert pid == os.getpid()
    
    def test_lock_file_format_validation(self, lock_file):
        """Test validation of lock file format."""
        # Create lock file with invalid format
        with open(lock_file, 'w') as f:
            f.write("invalid_format\n")
        
        # Should handle invalid format (treat as stale if old enough)
        lock = FileLock(str(lock_file))
        # Make file old
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(lock_file, (old_time, old_time))
        
        with patch('robust_tiff_compress.is_process_running', return_value=False):
            # Should be able to acquire after removing stale lock
            assert lock.acquire()


class TestLockRelease:
    """Test lock file release."""
    
    def test_lock_release_on_exit(self, lock_file):
        """Test that lock is released when exiting context manager."""
        with FileLock(str(lock_file)) as lock:
            assert lock.locked
            assert lock_file.exists()
        
        # After context exit, lock should be released
        assert not lock.locked
        assert not lock_file.exists()
    
    def test_lock_release_on_exception(self, lock_file):
        """Test that lock is released even when exception occurs."""
        try:
            with FileLock(str(lock_file)) as lock:
                assert lock.locked
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Lock should be released even after exception
        assert not lock_file.exists()
    
    def test_manual_lock_release(self, lock_file):
        """Test manually releasing a lock."""
        lock = FileLock(str(lock_file))
        lock.acquire()
        assert lock.locked
        assert lock_file.exists()
        
        lock.release()
        assert not lock.locked
        assert not lock_file.exists()
    
    def test_release_nonexistent_lock(self, lock_file):
        """Test releasing a lock that doesn't exist."""
        lock = FileLock(str(lock_file))
        lock.locked = True  # Simulate locked state
        
        # Should not raise error if file doesn't exist
        lock.release()
        assert not lock.locked


class TestConcurrentRunPrevention:
    """Test prevention of concurrent runs."""
    
    def test_runtime_error_when_lock_exists_and_process_running(
        self, existing_lock_file, mock_process_running
    ):
        """Test RuntimeError when lock exists and process is running."""
        # Set lock file to have PID of running process
        with open(existing_lock_file, 'w') as f:
            f.write("12345\n2024-01-01T00:00:00\n")
        
        with pytest.raises(RuntimeError, match="Another compression process"):
            with FileLock(str(existing_lock_file)):
                pass
    
    def test_lock_file_io_error_handling(self, lock_file):
        """Test handling of IO errors when creating lock file."""
        # Make directory read-only (if possible) or mock IOError
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            lock = FileLock(str(lock_file))
            assert not lock.acquire()
            assert not lock.locked
    
    def test_lock_file_removal_error_handling(self, lock_file):
        """Test handling of errors when removing lock file."""
        lock = FileLock(str(lock_file))
        lock.acquire()
        
        # Mock OSError when trying to remove
        with patch('os.remove', side_effect=OSError("Permission denied")):
            lock.release()
            # Should handle error gracefully
            assert not lock.locked  # State updated even if file removal fails


class TestLockFileEdgeCases:
    """Test edge cases for lock file handling."""
    
    def test_lock_file_with_missing_pid(self, lock_file):
        """Test handling of lock file with missing PID."""
        with open(lock_file, 'w') as f:
            f.write("2024-01-01T00:00:00\n")  # Missing PID line
        
        # Make file old
        old_time = time.time() - (25 * 3600)
        os.utime(lock_file, (old_time, old_time))
        
        lock = FileLock(str(lock_file))
        with patch('robust_tiff_compress.is_process_running', return_value=False):
            # Should handle gracefully and acquire lock
            assert lock.acquire()
    
    def test_lock_file_with_invalid_pid(self, lock_file):
        """Test handling of lock file with invalid PID."""
        with open(lock_file, 'w') as f:
            f.write("not_a_pid\n2024-01-01T00:00:00\n")
        
        # Make file old
        old_time = time.time() - (25 * 3600)
        os.utime(lock_file, (old_time, old_time))
        
        lock = FileLock(str(lock_file))
        with patch('robust_tiff_compress.is_process_running', return_value=False):
            # Should handle gracefully
            assert lock.acquire()
    
    def test_concurrent_lock_attempts(self, lock_file):
        """Test multiple attempts to acquire the same lock."""
        lock1 = FileLock(str(lock_file))
        lock2 = FileLock(str(lock_file))
        
        # First lock should succeed
        assert lock1.acquire()
        assert lock1.locked
        
        # Second lock should fail (same process, but lock already held)
        # Note: In real scenario, this would be different processes
        # For same process, the lock file exists check will prevent acquisition
        assert not lock2.acquire()
        assert not lock2.locked
        
        # Release first lock
        lock1.release()
        
        # Now second lock should succeed
        assert lock2.acquire()
        assert lock2.locked
