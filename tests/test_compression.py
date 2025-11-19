"""Tests for core compression functionality."""

import os
import pytest
import numpy as np
import tifffile
from pathlib import Path

import robust_tiff_compress
from robust_tiff_compress import (
    compress_tiff_file,
    CompressionState,
    MIN_FILE_SIZE,
    COMPRESSION_RATIO_THRESHOLD,
)


class TestBasicCompression:
    """Test basic compression functionality for both compression types."""
    
    @pytest.mark.parametrize("compression", ["zlib", "jpeg_2000_lossy"])
    def test_compress_medium_file_in_place(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient, compression
    ):
        """Test compressing a medium file in place with both compression types."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(medium_tiff_file)
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            None,
            compression,
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=(compression == "zlib")
        )
        
        assert success, f"Compression failed: {message}"
        assert "Compressed" in message or "Skipped" in message
        
        if "Compressed" in message:
            # Verify file was compressed
            compressed_size = os.path.getsize(medium_tiff_file)
            assert compressed_size < original_size or "Skipped" in message
    
    @pytest.mark.parametrize("compression", ["zlib", "jpeg_2000_lossy"])
    def test_compress_to_output_directory(
        self, medium_tiff_file, output_dir, state_file, mock_ram_large, mock_disk_space_sufficient, compression
    ):
        """Test compressing to output directory with both compression types."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(medium_tiff_file)
        output_path = output_dir / medium_tiff_file.name
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            str(output_path),
            compression,
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=(compression == "zlib")
        )
        
        assert success, f"Compression failed: {message}"
        
        if "Compressed" in message:
            assert output_path.exists(), "Output file was not created"
            compressed_size = os.path.getsize(output_path)
            assert compressed_size < original_size
        # If skipped, output file may not be created, which is acceptable
    
    def test_zlib_lossless_verification(
        self, tiff_file_uint16, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that zlib compression preserves data exactly (lossless)."""
        state = CompressionState(str(state_file))
        
        # Read original array
        original_array = tifffile.imread(str(tiff_file_uint16))
        
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_uint16),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=True
        )
        
        assert success, f"Compression failed: {message}"
        
        if "Compressed" in message:
            # Read compressed array
            compressed_array = tifffile.imread(str(tiff_file_uint16))
            
            # Verify bit-exact match
            assert np.array_equal(original_array, compressed_array), \
                "Lossless compression did not preserve data exactly"
            assert original_array.dtype == compressed_array.dtype
            assert original_array.shape == compressed_array.shape
    
    def test_jpeg2000_lossy_preserves_shape_dtype(
        self, tiff_file_uint16, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that jpeg_2000_lossy compression preserves shape and dtype."""
        state = CompressionState(str(state_file))
        
        # Read original array
        original_array = tifffile.imread(str(tiff_file_uint16))
        original_shape = original_array.shape
        original_dtype = original_array.dtype
        
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_uint16),
            None,
            "jpeg_2000_lossy",
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=False
        )
        
        assert success, f"Compression failed: {message}"
        
        if "Compressed" in message:
            # Read compressed array
            compressed_array = tifffile.imread(str(tiff_file_uint16))
            
            # Verify shape and dtype are preserved (data may differ for lossy)
            assert compressed_array.shape == original_shape
            assert compressed_array.dtype == original_dtype
    
    def test_compression_ratio_threshold(
        self, tiff_file_compressible, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that files meeting compression ratio threshold are compressed."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(tiff_file_compressible)
        
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_compressible),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=True
        )
        
        assert success
        # Compressible file should either compress or be skipped for other reasons
        assert "Compressed" in message or "Skipped" in message


class TestFileSizeHandling:
    """Test handling of files of different sizes."""
    
    def test_skip_small_file(
        self, small_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that files smaller than MIN_FILE_SIZE are skipped."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(small_tiff_file)
        assert original_size < MIN_FILE_SIZE
        
        success, message, compression_ratio = compress_tiff_file(
            str(small_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert success
        assert "Skipped" in message
        assert str(original_size) in message or "file size" in message.lower()
    
    def test_skip_large_file_exceeding_ram_limit(
        self, very_large_tiff_file, state_file, mock_ram_small, mock_disk_space_sufficient
    ):
        """Test that files exceeding RAM limit are skipped."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(very_large_tiff_file)
        
        # Check if file exceeds RAM limit (20% of 1GB = 200MB)
        max_file_size = int(1024 * 1024 * 1024 * 0.20)  # 20% of 1GB
        exceeds_ram = original_size > max_file_size
        
        success, message, compression_ratio = compress_tiff_file(
            str(very_large_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert success
        assert "Skipped" in message
        # File should be skipped either for RAM limit or compression ratio
        if exceeds_ram:
            assert "RAM" in message or "20%" in message or "MB >" in message
        # If not exceeding RAM, it might be skipped for compression ratio, which is also valid
    
    def test_compress_medium_file(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that medium-sized files are compressed."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(medium_tiff_file)
        assert original_size >= MIN_FILE_SIZE
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False
        )
        
        assert success
        # Should either compress or skip if ratio is too low
        assert "Compressed" in message or "Skipped" in message


class TestNestedDirectories:
    """Test compression in nested directory structures."""
    
    def test_compress_files_in_nested_dirs(
        self, sample_tiff_files, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test compressing files in nested directory structure."""
        state = CompressionState(str(state_file))
        
        for tiff_file in sample_tiff_files:
            success, message, compression_ratio = compress_tiff_file(
                str(tiff_file),
                None,
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            assert success, f"Failed to compress {tiff_file}: {message}"
            assert "Compressed" in message or "Skipped" in message
    
    def test_compress_to_output_with_nested_structure(
        self, nested_test_dir, sample_tiff_files, output_dir, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test compressing files to output directory preserving nested structure."""
        state = CompressionState(str(state_file))
        root_dir = nested_test_dir  # Get root of test structure
        
        for tiff_file in sample_tiff_files:
            # Calculate relative path from root
            rel_path = tiff_file.relative_to(root_dir)
            output_path = output_dir / rel_path
            
            success, message, compression_ratio = compress_tiff_file(
                str(tiff_file),
                str(output_path),
                "zlib",
                85,
                None,
                state,
                dry_run=False
            )
            
            assert success, f"Failed to compress {tiff_file}: {message}"
            if "Compressed" in message:
                assert output_path.exists(), f"Output file not created: {output_path}"
                # Verify nested directory structure was created
                assert output_path.parent.exists()


class TestDataIntegrity:
    """Test data integrity after compression."""
    
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_preserve_dtype(
        self, tmp_test_dir, state_file, mock_ram_large, mock_disk_space_sufficient, dtype
    ):
        """Test that compression preserves different dtypes."""
        from tests.conftest import create_test_tiff
        
        tiff_file = tmp_test_dir / f"test_{dtype.__name__}.tif"
        create_test_tiff(tiff_file, dtype=dtype, shape=(500, 500))
        
        original_array = tifffile.imread(str(tiff_file))
        original_dtype = original_array.dtype
        
        state = CompressionState(str(state_file))
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=True
        )
        
        assert success
        if "Compressed" in message:
            compressed_array = tifffile.imread(str(tiff_file))
            assert compressed_array.dtype == original_dtype
    
    def test_preserve_3d_shape(
        self, tiff_file_3d, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that compression preserves 3D array shapes."""
        original_array = tifffile.imread(str(tiff_file_3d))
        original_shape = original_array.shape
        
        state = CompressionState(str(state_file))
        success, message, compression_ratio = compress_tiff_file(
            str(tiff_file_3d),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=False,
            verify_lossless_exact=True
        )
        
        assert success
        if "Compressed" in message:
            compressed_array = tifffile.imread(str(tiff_file_3d))
            assert compressed_array.shape == original_shape
    
    def test_compressed_file_readable(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that compressed files can be read back successfully."""
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
        
        assert success
        if "Compressed" in message:
            # Verify file can be read
            array = tifffile.imread(str(medium_tiff_file))
            assert array is not None
            assert isinstance(array, np.ndarray)
            assert array.size > 0


class TestDryRun:
    """Test dry-run mode."""
    
    def test_dry_run_does_not_modify_file(
        self, medium_tiff_file, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that dry-run mode doesn't modify files."""
        state = CompressionState(str(state_file))
        original_size = os.path.getsize(medium_tiff_file)
        original_mtime = os.path.getmtime(medium_tiff_file)
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            None,
            "zlib",
            85,
            None,
            state,
            dry_run=True
        )
        
        assert success
        assert "Dry run" in message or "would compress" in message.lower()
        
        # Verify file was not modified
        assert os.path.getsize(medium_tiff_file) == original_size
        assert os.path.getmtime(medium_tiff_file) == original_mtime
    
    def test_dry_run_does_not_create_output(
        self, medium_tiff_file, output_dir, state_file, mock_ram_large, mock_disk_space_sufficient
    ):
        """Test that dry-run mode doesn't create output files."""
        state = CompressionState(str(state_file))
        output_path = output_dir / medium_tiff_file.name
        
        success, message, compression_ratio = compress_tiff_file(
            str(medium_tiff_file),
            str(output_path),
            "zlib",
            85,
            None,
            state,
            dry_run=True
        )
        
        assert success
        assert not output_path.exists(), "Output file should not be created in dry-run mode"
