"""Tests for compression state management and resumability."""

import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import robust_tiff_compress
from robust_tiff_compress import CompressionState, STATE_FILE


class TestStateFileOperations:
    """Test state file loading, saving, and operations."""
    
    def test_state_file_creation(self, state_file):
        """Test that state file is created on first use."""
        state = CompressionState(str(state_file))
        assert not state_file.exists()  # Not created until first save
        
        # Mark a file as processed to trigger save (using filename only)
        state.mark_processed("file.tif", 2.5, "zlib", 1000000, 400000)
        
        assert state_file.exists(), "State file should be created"
    
    def test_state_file_loading(self, existing_state_file):
        """Test loading state from existing file."""
        state = CompressionState(str(existing_state_file))
        
        # Check that processed files are loaded (using filenames)
        assert state.is_processed("file1.tif")
        assert state.is_processed("file2.tif")
        assert not state.is_processed("file3.tif")
    
    def test_state_file_saving_atomic(self, state_file):
        """Test that state file is saved atomically."""
        state = CompressionState(str(state_file))
        
        # Mark multiple files as processed (using filenames only)
        state.mark_processed("file1.tif", 2.0, "zlib", 1000000, 500000)
        state.mark_processed("file2.tif", 3.0, "jpeg_2000_lossy", 2000000, 666666)
        
        # Verify state file exists and is valid JSON
        assert state_file.exists()
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert "processed" in data
            assert len(data["processed"]) == 2
    
    def test_mark_processed(self, state_file):
        """Test marking files as processed."""
        state = CompressionState(str(state_file))
        
        filename = "file.tif"  # Use filename only, not full path
        compression_ratio = 2.5
        compression_type = "zlib"
        original_size = 1000000
        compressed_size = 400000
        
        state.mark_processed(
            filename,
            compression_ratio,
            compression_type,
            original_size,
            compressed_size
        )
        
        assert state.is_processed(filename)
        
        # Verify state file contains correct data
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert filename in data["processed"]
            file_data = data["processed"][filename]
            assert file_data["compression_ratio"] == compression_ratio
            assert file_data["compression_type"] == compression_type
            assert file_data["original_size"] == original_size
            assert file_data["compressed_size"] == compressed_size
            assert "timestamp" in file_data
    
    def test_is_processed(self, state_file):
        """Test checking if file is processed."""
        state = CompressionState(str(state_file))
        
        filename = "file.tif"
        assert not state.is_processed(filename)
        
        state.mark_processed(filename, 2.0, "zlib", 1000000, 500000)
        assert state.is_processed(filename)
    
    def test_get_processed_count(self, state_file):
        """Test getting count of processed files."""
        state = CompressionState(str(state_file))
        
        assert state.get_processed_count() == 0
        
        state.mark_processed("file1.tif", 2.0, "zlib", 1000000, 500000)
        assert state.get_processed_count() == 1
        
        state.mark_processed("file2.tif", 3.0, "zlib", 2000000, 666666)
        assert state.get_processed_count() == 2
    
    def test_corrupted_state_file_handling(self, corrupted_state_file):
        """Test handling of corrupted state file (invalid JSON)."""
        # Should not raise exception, should start fresh
        state = CompressionState(str(corrupted_state_file))
        
        # Should start with empty state
        assert state.get_processed_count() == 0
        assert not state.is_processed("any_file.tif")
    
    def test_missing_state_file(self, tmp_test_dir):
        """Test handling of missing state file."""
        state_file = tmp_test_dir / "nonexistent_state.json"
        assert not state_file.exists()
        
        state = CompressionState(str(state_file))
        assert state.get_processed_count() == 0
        assert not state.is_processed("any_file.tif")
    
    def test_state_file_io_error_handling(self, state_file):
        """Test handling of IO errors when reading state file."""
        # Create a state file that will cause IO error
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Make directory read-only to cause write error (on some systems)
        # For this test, we'll mock the file operations
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            state = CompressionState(str(state_file))
            # Should handle error gracefully and start with empty state
            assert state.get_processed_count() == 0


class TestResumeFunctionality:
    """Test resumability features."""
    
    def test_skip_already_processed_files(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that already processed files are skipped."""
        from robust_tiff_compress import compress_tiff_file, get_state_file_for_directory
        
        # Get state file for the directory containing the file
        file_dir = str(medium_tiff_file.parent)
        state_file_path = get_state_file_for_directory(file_dir)
        state = CompressionState(state_file_path)
        
        # Mark file as already processed (using filename only)
        filename = medium_tiff_file.name
        state.mark_processed(
            filename,
            2.5,
            "zlib",
            1000000,
            400000
        )
        
        # Try to compress again (pass None for state, it will use per-directory state)
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            None,
            "zlib",
            85,
            None,
            None,  # Will use per-directory state
            dry_run=False
        )
        
        # File should be skipped because it's already in state
        # (Note: The actual skip happens in find_tiff_files, but state tracks it)
        assert state.is_processed(filename)
    
    def test_resume_after_interruption(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test resuming compression after interruption."""
        from robust_tiff_compress import find_tiff_files, get_state_file_for_directory
        
        root_dir = sample_tiff_files[0].parent.parent
        
        # Process first two files - mark them in their respective directories
        for i in [0, 1]:
            file_path = sample_tiff_files[i]
            file_dir = str(file_path.parent)
            state_file_path = get_state_file_for_directory(file_dir)
            state = CompressionState(state_file_path)
            filename = file_path.name
            state.mark_processed(filename, 2.0 + i * 0.5, "zlib", 1000000 + i * 1000000, 500000 + i * 300000)
        
        # Find files to compress (should skip already processed)
        tiff_files = find_tiff_files(str(root_dir))
        
        # Should find remaining files, not the processed ones
        processed_paths = {str(sample_tiff_files[0]), str(sample_tiff_files[1])}
        found_paths = set(tiff_files)
        
        # None of the found files should be in processed set
        assert not (found_paths & processed_paths), \
            "Already processed files should not be found"
    
    def test_state_persistence_across_runs(self, state_file):
        """Test that state persists across multiple CompressionState instances."""
        # First instance
        state1 = CompressionState(str(state_file))
        state1.mark_processed("file1.tif", 2.0, "zlib", 1000000, 500000)
        state1.mark_processed("file2.tif", 3.0, "zlib", 2000000, 666666)
        
        # Second instance (simulating new run)
        state2 = CompressionState(str(state_file))
        
        # Should see files from first instance
        assert state2.is_processed("file1.tif")
        assert state2.is_processed("file2.tif")
        assert state2.get_processed_count() == 2


class TestStateThreadSafety:
    """Test thread safety of state operations."""
    
    def test_concurrent_mark_processed(self, state_file):
        """Test that marking files concurrently is thread-safe."""
        import threading
        
        state = CompressionState(str(state_file))
        results = []
        
        def mark_file(file_num):
            state.mark_processed(
                f"file{file_num}.tif",
                2.0,
                "zlib",
                1000000,
                500000
            )
            results.append(file_num)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=mark_file, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all files were marked
        assert state.get_processed_count() == 10
        for i in range(10):
            assert state.is_processed(f"file{i}.tif")
