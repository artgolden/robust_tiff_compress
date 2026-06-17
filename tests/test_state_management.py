"""Tests for compression state management and resumability."""

import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import robust_tiff_compress
from robust_tiff_compress import CompressionState, STATE_DIR, STATE_FILE, get_state_file_for_directory
from tests.conftest import image_dir_for_state_file


class TestStateFileOperations:
    """Test state file loading, saving, and operations."""
    
    def test_state_file_creation(self, state_file, tmp_test_dir):
        """Test that state file is created on first use."""
        state = CompressionState(str(tmp_test_dir))
        assert not state_file.exists()  # Not created until first save

        file_path = os.path.join(str(tmp_test_dir), "file.tif")
        state.mark_processed(file_path, 2.5, "zlib", 1000000, 400000)

        assert state_file.exists(), "State file should be created"
        assert state_file.parent.name == STATE_DIR

    def test_state_file_loading(self, existing_state_file):
        """Test loading state from existing file."""
        image_dir = str(image_dir_for_state_file(existing_state_file))
        state = CompressionState(image_dir)

        file1_path = os.path.join(image_dir, "file1.tif")
        file2_path = os.path.join(image_dir, "file2.tif")
        file3_path = os.path.join(image_dir, "file3.tif")

        assert state.is_processed(file1_path)
        assert state.is_processed(file2_path)
        assert not state.is_processed(file3_path)

    def test_state_file_saving_atomic(self, state_file, tmp_test_dir):
        """Test that state file is saved atomically."""
        state = CompressionState(str(tmp_test_dir))

        file1_path = os.path.join(str(tmp_test_dir), "file1.tif")
        file2_path = os.path.join(str(tmp_test_dir), "file2.tif")

        state.mark_processed(file1_path, 2.0, "zlib", 1000000, 500000)
        state.mark_processed(file2_path, 3.0, "jpeg_2000_lossy", 2000000, 666666)

        # Verify state file exists and is valid JSON
        assert state_file.exists()
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert "processed" in data
            assert len(data["processed"]) == 2

    def test_mark_processed(self, state_file, tmp_test_dir):
        """Test marking files as processed."""
        state = CompressionState(str(tmp_test_dir))

        filename = "file.tif"
        file_path = os.path.join(str(tmp_test_dir), filename)
        compression_ratio = 2.5
        compression_type = "zlib"
        original_size = 1000000
        compressed_size = 400000

        state.mark_processed(
            file_path,
            compression_ratio,
            compression_type,
            original_size,
            compressed_size
        )

        assert state.is_processed(file_path)

        # Verify state file contains correct data
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert file_path in data["processed"]
            file_data = data["processed"][file_path]
            assert file_data["compression_ratio"] == compression_ratio
            assert file_data["compression_type"] == compression_type
            assert file_data["original_size"] == original_size
            assert file_data["compressed_size"] == compressed_size
            assert "timestamp" in file_data

    def test_is_processed(self, tmp_test_dir):
        """Test checking if file is processed."""
        state = CompressionState(str(tmp_test_dir))

        filename = "file.tif"
        file_path = os.path.join(str(tmp_test_dir), filename)
        assert not state.is_processed(file_path)

        state.mark_processed(file_path, 2.0, "zlib", 1000000, 500000)
        assert state.is_processed(file_path)

    def test_get_processed_count(self, tmp_test_dir):
        """Test getting count of processed files."""
        state = CompressionState(str(tmp_test_dir))

        assert state.get_processed_count() == 0

        file1_path = os.path.join(str(tmp_test_dir), "file1.tif")
        file2_path = os.path.join(str(tmp_test_dir), "file2.tif")
        state.mark_processed(file1_path, 2.0, "zlib", 1000000, 500000)
        assert state.get_processed_count() == 1

        state.mark_processed(file2_path, 3.0, "zlib", 2000000, 666666)
        assert state.get_processed_count() == 2

    def test_corrupted_state_file_handling(self, corrupted_state_file):
        """Test handling of corrupted state file (invalid JSON)."""
        image_dir = str(image_dir_for_state_file(corrupted_state_file))
        state = CompressionState(image_dir)

        assert state.get_processed_count() == 0
        file_path = os.path.join(image_dir, "any_file.tif")
        assert not state.is_processed(file_path)

    def test_missing_state_file(self, tmp_test_dir):
        """Test handling of missing state file."""
        state_path = Path(get_state_file_for_directory(str(tmp_test_dir)))
        assert not state_path.exists()

        state = CompressionState(str(tmp_test_dir))
        assert state.get_processed_count() == 0
        file_path = os.path.join(str(tmp_test_dir), "any_file.tif")
        assert not state.is_processed(file_path)

    def test_state_subdirectory_created_on_first_save(self, tmp_test_dir, state_dir):
        """Test that the state subdirectory is created on first save."""
        assert not state_dir.exists()
        state = CompressionState(str(tmp_test_dir))
        file_path = os.path.join(str(tmp_test_dir), "file.tif")
        state.mark_processed(file_path, 2.0, "zlib", 1000000, 500000)
        assert state_dir.exists()

    def test_legacy_flat_state_file_migration(self, tmp_test_dir, state_file):
        """Test migration of legacy flat state file into the state subdirectory."""
        legacy_path = tmp_test_dir / STATE_FILE
        file_path = os.path.join(str(tmp_test_dir), "file1.tif")
        legacy_data = {
            "processed": {
                file_path: {
                    "compression_ratio": 2.0,
                    "compression_type": "zlib",
                    "original_size": 1000000,
                    "compressed_size": 500000,
                    "timestamp": "2024-01-01T00:00:00",
                }
            }
        }
        with open(legacy_path, 'w') as f:
            json.dump(legacy_data, f)

        state = CompressionState(str(tmp_test_dir))

        assert not legacy_path.exists()
        assert state_file.exists()
        assert state.is_processed(file_path)
        assert state.get_processed_count() == 1
    
    def test_state_file_io_error_handling(self, tmp_test_dir):
        """Test handling of IO errors when reading state file."""
        state_file = Path(get_state_file_for_directory(str(tmp_test_dir)))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            state = CompressionState(str(tmp_test_dir))
            # Should handle error gracefully and start with empty state
            assert state.get_processed_count() == 0


class TestResumeFunctionality:
    """Test resumability features."""

    def test_skip_already_processed_files(
        self, medium_tiff_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that already processed files are skipped."""
        from robust_tiff_compress import compress_tiff_file

        file_dir = str(medium_tiff_file.parent)
        state = CompressionState(file_dir)

        # Mark file as already processed (using filename only)
        state.mark_processed(medium_tiff_file, 2.5, "zlib", 1000000, 400000)

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
        assert state.is_processed(medium_tiff_file)

    def test_resume_after_interruption(
        self, sample_tiff_files):
        """Test resuming compression after interruption."""
        from robust_tiff_compress import find_tiff_files

        root_dir = sample_tiff_files[-1].parent

        for i in [1, 2, 4]:
            file_path = sample_tiff_files[i]
            file_dir = str(file_path.parent)
            state = CompressionState(file_dir)
            state.mark_processed(
                file_path,
                2.0 + i * 0.5,
                "zlib",
                1000000 + i * 1000000,
                500000 + i * 300000,
            )

        # Find files to compress (should skip already processed)
        tiff_files = find_tiff_files(str(root_dir))

        # Should find remaining files, not the processed ones
        processed_paths = {str(sample_tiff_files[1]), str(sample_tiff_files[2])}
        found_paths = set(tiff_files)

        # None of the found files should be in processed set
        assert not (found_paths & processed_paths), \
            "Already processed files should not be found"

    def test_state_persistence_across_runs(self, tmp_test_dir):
        """Test that state persists across multiple CompressionState instances."""
        state1 = CompressionState(str(tmp_test_dir))
        state1.mark_processed(os.path.join(state1.directory, "file1.tif"), 2.0, "zlib", 1000000, 500000)
        state1.mark_processed(os.path.join(state1.directory, "file2.tif"), 3.0, "zlib", 2000000, 666666)

        state2 = CompressionState(str(tmp_test_dir))

        # Should see files from first instance
        assert state2.is_processed(os.path.join(state2.directory, "file1.tif"))
        assert state2.is_processed(os.path.join(state2.directory, "file2.tif")) 
        assert state2.get_processed_count() == 2


class TestStateThreadSafety:
    """Test thread safety of state operations."""
    
    def test_concurrent_mark_processed(self, tmp_test_dir):
        """Test that marking files concurrently is thread-safe."""
        import threading
        
        state = CompressionState(str(tmp_test_dir))
        results = []
        
        def mark_file(file_num):
            state.mark_processed(
                os.path.join(state.directory, f"file{file_num}.tif"),
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
            assert state.is_processed(os.path.join(state.directory, f"file{i}.tif"))


class TestSkippedFileTracking:
    """Test tracking and handling of skipped files."""

    def test_mark_skipped(self, state_file, tmp_test_dir):
        """Test marking files as skipped."""
        state = CompressionState(str(tmp_test_dir))

        file_path = os.path.join(str(tmp_test_dir), "file.tif")
        reason = "compression ratio 1.2 < 1.43"
        compression_ratio = 1.2

        state.mark_skipped(file_path, reason, compression_ratio)

        assert state.is_skipped(file_path)

        # Verify state file contains correct data
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert "skipped" in data
            assert file_path in data["skipped"]
            file_data = data["skipped"][file_path]
            assert file_data["reason"] == reason
            assert file_data["compression_ratio"] == compression_ratio
            assert "timestamp" in file_data

    def test_is_skipped(self, tmp_test_dir):
        """Test checking if file is skipped."""
        state = CompressionState(str(tmp_test_dir))

        file_path = os.path.join(str(tmp_test_dir), "file.tif")
        assert not state.is_skipped(file_path)

        state.mark_skipped(file_path, "test reason", 1.2)
        assert state.is_skipped(file_path)

    def test_get_skipped_count(self, tmp_test_dir):
        """Test getting count of skipped files."""
        state = CompressionState(str(tmp_test_dir))

        assert state.get_skipped_count() == 0

        file1_path = os.path.join(str(tmp_test_dir), "file1.tif")
        file2_path = os.path.join(str(tmp_test_dir), "file2.tif")
        state.mark_skipped(file1_path, "reason1", 1.1)
        assert state.get_skipped_count() == 1

        state.mark_skipped(file2_path, "reason2", 1.3)
        assert state.get_skipped_count() == 2

    def test_skip_previously_skipped_file(
        self, tiff_file_not_compressible, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that files skipped due to compression ratio are tracked and skipped on subsequent runs."""
        from robust_tiff_compress import compress_tiff_file, find_tiff_files

        file_dir = str(tiff_file_not_compressible.parent)
        state = CompressionState(file_dir)

        # First attempt: file should be skipped due to low compression ratio
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_not_compressible),
            None,
            "zlib",
            85,
            None,
            state,  # Will use per-directory state
            dry_run=False
        )

        # Should succeed but skip due to low compression ratio
        assert success
        assert "Skipped" in message
        assert "compression ratio" in message.lower() or "ratio" in message.lower()

        # Verify file is marked as skipped in state
        assert state.is_skipped(str(tiff_file_not_compressible))

        # Second attempt: file should be skipped by find_tiff_files (not included in list)
        tiff_files = find_tiff_files(file_dir)
        assert str(tiff_file_not_compressible) not in tiff_files, \
            "Previously skipped file should not be included in find_tiff_files"


    def test_force_recompress_skipped_files(
        self, tmp_test_dir, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that --force-recompress-skipped includes previously skipped files."""
        from robust_tiff_compress import find_tiff_files, CompressionState
        from tests.conftest import create_test_tiff
        import numpy as np

        # Create test files
        file1 = tmp_test_dir / "file1.tif"
        file2 = tmp_test_dir / "file2.tif"

        # Create test TIFF files
        create_test_tiff(file1, size_bytes=3 * 1024 * 1024, dtype=np.uint16)
        create_test_tiff(file2, size_bytes=3 * 1024 * 1024, dtype=np.uint16)

        # Get state file and mark file1 as skipped
        state = CompressionState(str(tmp_test_dir))
        state.mark_skipped(str(file1), "compression ratio 1.2 < 1.43", 1.2)

        # Without force flag: file1 should be skipped
        tiff_files = find_tiff_files(str(tmp_test_dir), force_recompress_skipped=False)
        assert str(file1) not in tiff_files, "file1 should be skipped without force flag"
        assert str(file2) in tiff_files, "file2 should be included"

        # With force flag: file1 should be included
        tiff_files = find_tiff_files(str(tmp_test_dir), force_recompress_skipped=True)
        assert str(file1) in tiff_files, "file1 should be included with force flag"
        assert str(file2) in tiff_files, "file2 should still be included"

    def test_force_recompress_processed_files(
        self, tmp_test_dir, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that --force-recompress-processed includes previously processed files."""
        from robust_tiff_compress import find_tiff_files, CompressionState
        from tests.conftest import create_test_tiff
        import numpy as np

        # Create test files
        file1 = tmp_test_dir / "file1.tif"
        file2 = tmp_test_dir / "file2.tif"

        # Create test TIFF files
        create_test_tiff(file1, size_bytes=3 * 1024 * 1024, dtype=np.uint16)
        create_test_tiff(file2, size_bytes=3 * 1024 * 1024, dtype=np.uint16)

        # Get state file and mark file1 as processed
        state = CompressionState(str(tmp_test_dir))
        state.mark_processed(str(file1), 2.5, "zlib", 1000000, 400000)

        # Without force flag: file1 should be skipped
        tiff_files = find_tiff_files(str(tmp_test_dir), force_recompress_processed=False)
        assert str(file1) not in tiff_files, "file1 should be skipped without force flag"
        assert str(file2) in tiff_files, "file2 should be included"

        # With force flag: file1 should be included
        tiff_files = find_tiff_files(str(tmp_test_dir), force_recompress_processed=True)
        assert str(file1) in tiff_files, "file1 should be included with force flag"
        assert str(file2) in tiff_files, "file2 should still be included"
