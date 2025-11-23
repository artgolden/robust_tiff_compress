#!/usr/bin/env python3
"""
Robust TIFF compression tool for TrueNAS Scale storage servers.

This tool compresses TIFF files in a directory tree with focus on stability,
reliability, and proper handling of edge cases.

IMPORTANT: This tool is designed to work ONLY on local filesystems. It does NOT
support network-attached storage (NAS), SMB/CIFS shares, or other remote filesystems.
Using this tool on network filesystems may result in data corruption or other errors.
"""

import os
import sys
import json
import argparse
import shutil
import time
import logging
import multiprocessing
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Callable
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import pkg_resources
import tifffile
import numpy as np
from tqdm import tqdm


# Constants
MIN_FILE_SIZE = 1024 * 1024  # 1MB in bytes
COMPRESSION_RATIO_THRESHOLD = 1.43  # At least 30% reduction (1/1.43 ≈ 0.70, so 30% reduction minimum)
STATE_FILE = "_compression_state.json"
TEMP_SUFFIX = ".compressing"
TEMP_ERROR_SUFFIX = ".compressing.ERROR"
RAM_SIZE_LIMIT_RATIO = 0.40  # Don't compress files >40% of free RAM
LOCK_FILE = "_compression_lock"
WARNING_FILE = "_compression_stopped_warning.txt"
MAX_CONSECUTIVE_ERRORS = 3


def get_file_ownership(file_path: str) -> Optional[Tuple[int, int, int]]:
    """
    Get file ownership (uid, gid) and permissions (mode).
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (uid, gid, mode) if successful, None otherwise
    """
    try:
        stat_info = os.stat(file_path)
        return (stat_info.st_uid, stat_info.st_gid, stat_info.st_mode)
    except OSError:
        return None


def set_file_ownership(file_path: str, uid: int, gid: int, mode: Optional[int] = None) -> bool:
    """
    Set file ownership (uid, gid) and optionally permissions (mode).
    
    Args:
        file_path: Path to the file
        uid: User ID
        gid: Group ID
        mode: Optional file mode (permissions). If None, only ownership is set.
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.chown(file_path, uid, gid)
        if mode is not None:
            os.chmod(file_path, mode)
        return True
    except (OSError, PermissionError) as e:
        logging.warning(f"Failed to set ownership/permissions for {file_path}: {e}")
        return False


def get_directory_ownership(directory: str) -> Optional[Tuple[int, int, int]]:
    """
    Get directory ownership (uid, gid) and permissions (mode).
    
    Args:
        directory: Path to the directory
        
    Returns:
        Tuple of (uid, gid, mode) if successful, None otherwise
    """
    try:
        stat_info = os.stat(directory)
        return (stat_info.st_uid, stat_info.st_gid, stat_info.st_mode)
    except OSError:
        return None


def set_state_file_ownership(state_file: str) -> bool:
    """
    Set STATE_FILE ownership to match its directory and set read+write permissions for group,
    and read for others (rw-rw-r--).
    
    Args:
        state_file: Path to the state file
        
    Returns:
        True if successful, False otherwise
    """
    state_dir = os.path.dirname(state_file)
    if not state_dir:
        state_dir = '.'
    
    # Get directory ownership
    dir_ownership = get_directory_ownership(state_dir)
    if dir_ownership is None:
        return False
    
    uid, gid, _ = dir_ownership
    
    # Set ownership to match directory
    # Set permissions to read+write for owner and group, read for others (0o664 = rw-rw-r--)
    return set_file_ownership(state_file, uid, gid, 0o664)


class CompressionState:
    """Manages compression state for resumability per directory."""

    def __init__(self, state_file: str, preserve_ownership: bool = False):
        self.state_file = state_file
        self.directory = os.path.dirname(state_file)
        self.state: Dict = {}
        self.lock = threading.Lock()
        self.preserve_ownership = preserve_ownership
        self._load()

    def _load(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load state file {self.state_file}: {e}. Starting fresh.")
                self.state = {}
        else:
            self.state = {}

    def _save(self):
        """Save state to file atomically."""
        # Ensure directory exists
        state_dir = os.path.dirname(self.state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        temp_file = self.state_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            # Atomic move on POSIX systems
            os.replace(temp_file, self.state_file)
            
            # Set ownership and permissions if preserve_ownership is enabled
            if self.preserve_ownership:
                set_state_file_ownership(self.state_file)
        except Exception as e:
            logging.error(f"Failed to save state file {self.state_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def is_processed(self, file_path: str) -> bool:
        """Check if file has been processed (by filename only, not full path)."""
        f_path = str(file_path)
        if not f_path.startswith(self.directory):
            raise ValueError(f"File path {f_path} is not in directory {self.directory}")
        with self.lock:
            return f_path in self.state.get("processed", {})

    def mark_processed(self, file_path: str, compression_ratio: float, 
                      compression_type: str, original_size: int, compressed_size: int):
        """Mark file as processed (by filename only, not full path)."""
        f_path = str(file_path)
        if not f_path.startswith(self.directory):
            raise ValueError(f"File path {f_path} is not in directory {self.directory}")
        with self.lock:
            if 'processed' not in self.state:
                self.state['processed'] = {}
            self.state['processed'][f_path] = {
                'compression_ratio': compression_ratio,
                'compression_type': compression_type,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'timestamp': datetime.now().isoformat()
            }
            self._save()

    def get_processed_count(self) -> int:
        """Get count of processed files."""
        with self.lock:
            return len(self.state.get('processed', {}))

    def is_skipped(self, file_path: str) -> bool:
        """Check if file has been skipped (by filename only, not full path)."""
        f_path = str(file_path)
        if not f_path.startswith(self.directory):
            raise ValueError(f"File path {f_path} is not in directory {self.directory}")
        with self.lock:
            return f_path in self.state.get("skipped", {})

    def mark_skipped(self, file_path: str, reason: str, compression_ratio: Optional[float] = None):
        """Mark file as skipped (by filename only, not full path)."""
        f_path = str(file_path)
        if not f_path.startswith(self.directory):
            raise ValueError(f"File path {f_path} is not in directory {self.directory}")
        with self.lock:
            if 'skipped' not in self.state:
                self.state['skipped'] = {}
            self.state['skipped'][f_path] = {
                'reason': reason,
                'compression_ratio': compression_ratio,
                'timestamp': datetime.now().isoformat()
            }
            self._save()

    def get_skipped_count(self) -> int:
        """Get count of skipped files."""
        with self.lock:
            return len(self.state.get('skipped', {}))


class FileLock:
    """Simple file-based lock for preventing concurrent runs."""
    
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.locked = False
    
    def acquire(self) -> bool:
        """Acquire lock."""
        if os.path.exists(self.lock_file):
            # Check if lock is stale (older than 24 hours) or process is not running
            try:
                lock_age = time.time() - os.path.getmtime(self.lock_file)
                # Read PID from lock file
                try:
                    with open(self.lock_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line.isdigit():
                            lock_pid = int(first_line)
                            # Check if process is still running
                            if is_process_running(lock_pid):
                                # Process is running, lock is valid
                                return False
                            else:
                                # Process is not running, lock is stale
                                logging.warning(
                                    f"Removing stale lock file: process {lock_pid} is not running "
                                    f"(lock age: {lock_age/3600:.1f} hours)"
                                )
                                os.remove(self.lock_file)
                        else:
                            # Invalid lock file format, check age
                            if lock_age > 86400:  # 24 hours
                                logging.warning(f"Removing stale lock file (age: {lock_age/3600:.1f} hours)")
                                os.remove(self.lock_file)
                            else:
                                return False
                except (ValueError, IOError) as e:
                    # Lock file exists but can't read it, check age
                    if lock_age > 86400:  # 24 hours
                        logging.warning(f"Removing stale lock file (age: {lock_age/3600:.1f} hours, read error: {e})")
                        os.remove(self.lock_file)
                    else:
                        return False
            except OSError:
                pass
        
        try:
            with open(self.lock_file, 'w') as f:
                f.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.locked = True
            return True
        except IOError:
            return False
    
    def release(self):
        """Release lock."""
        if self.locked:
            if os.path.exists(self.lock_file):
                try:
                    os.remove(self.lock_file)
                except OSError:
                    pass
            # Always update state, even if file removal fails
            self.locked = False
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Another compression process is running or lock file exists")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def verify_tiff_file(file_path: str) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Verify that a TIFF file can be opened and read as a numpy array.
    Returns (success, array or None).
    """
    try:
        array = tifffile.imread(file_path)
        if array is None :
            return False, None
        if not isinstance(array, np.ndarray) or array.size == 0:
            return False, None
        return True, array
    except Exception as e:
        logging.debug(f"Could not read {file_path}: {e}")
        return False, None


def get_available_ram() -> int:
    """Get free/available RAM in bytes.
    
    Returns the amount of RAM that is currently free and available for use.
    On Linux, this includes memory that can be freed by the OS (buffers/cache).
    
    Raises:
        RuntimeError: If RAM detection fails (no fallback to assumed value).
    """
    if PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().available
        except Exception as e:
            raise RuntimeError(f"Failed to detect free RAM using psutil: {e}")
    
    # Fallback: read from /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # Convert from KB to bytes
        raise RuntimeError("MemAvailable not found in /proc/meminfo")
    except Exception as e:
        raise RuntimeError(f"Failed to detect free RAM from /proc/meminfo: {e}")


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False
    
    # Fallback: use os.kill with signal 0 (doesn't kill, just checks)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def check_disk_space(path: str, required_bytes: int) -> Tuple[bool, int]:
    """
    Check if there's enough disk space at the given path.
    
    Returns:
        (has_space, available_bytes)
    """
    if PSUTIL_AVAILABLE:
        try:
            stat = psutil.disk_usage(path)
            return stat.free >= required_bytes, stat.free
        except Exception:
            pass
    
    # Fallback: use shutil.disk_usage (Python 3.3+)
    try:
        stat = shutil.disk_usage(path)
        return stat.free >= required_bytes, stat.free
    except (AttributeError, OSError):
        # Fallback: use statvfs (POSIX)
        try:
            if hasattr(os, 'statvfs'):
                stat = os.statvfs(path)
                free_bytes = stat.f_bavail * stat.f_frsize
                return free_bytes >= required_bytes, free_bytes
        except (AttributeError, OSError):
            pass
    
    # If we can't check, assume there's space (better than failing)
    logging.warning(f"Could not check disk space at {path}, assuming sufficient space")
    return True, 0


def arrays_equal_exact(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    Compare two numpy arrays for bit-exact equality (for lossless compression).
    Uses np.array_equal which is the most performant method for exact comparison.
    """
    try:
        if arr1.shape != arr2.shape:
            return False
        if arr1.dtype != arr2.dtype:
            return False
        return np.array_equal(arr1, arr2)
    except Exception:
        return False


def rename_temp_file_on_error(temp_path: str, file_path: str, error_msg: str) -> None:
    """
    Rename temporary file to error-marked name to exclude from cleanup.
    Notifies user via logging.
    
    Args:
        temp_path: Path to temporary file
        file_path: Original file path for logging
        error_msg: Error message for logging
    """
    if os.path.exists(temp_path):
        error_path = temp_path.replace(TEMP_SUFFIX, TEMP_ERROR_SUFFIX)
        try:
            os.rename(temp_path, error_path)
            logging.error(
                f"Error encountered for {file_path}: {error_msg}. "
                f"Temporary file renamed to {error_path} and excluded from cleanup. "
                f"Please investigate manually."
            )
        except Exception as rename_error:
            # If rename fails, log error but don't fail completely
            logging.error(
                f"Error encountered for {file_path}: {error_msg}. "
                f"Failed to rename temp file to error-marked name ({rename_error}). "
                f"Temp file may still exist at {temp_path}."
            )


def save_backup_from_array(array: np.ndarray, original_path: str) -> Optional[str]:
    """
    Save a backup copy of the original file from the in-memory array.
    
    Args:
        array: The numpy array containing the original image data
        original_path: Path to the original file (used to generate backup path)
    
    Returns:
        Path to the backup file if successful, None otherwise
    """
    try:
        # Generate backup path: original_path + timestamp + .backup.tif
        base_path = os.path.splitext(original_path)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{base_path}.backup_{timestamp}.tif"
        
        # Ensure backup path doesn't exist (unlikely but possible)
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{base_path}.backup_{timestamp}_{counter}.tif"
            counter += 1
        
        # Write the array to the backup file
        with tifffile.TiffWriter(backup_path) as tiff:
            tiff.write(array)
        
        # Verify backup was created
        if os.path.exists(backup_path) and os.path.getsize(backup_path) > 0:
            logging.critical(
                f"Saved backup of original file to {backup_path} "
                f"(size: {os.path.getsize(backup_path) / (1024**2):.1f} MB)"
            )
            return backup_path
        else:
            logging.error(f"Backup file was created but appears to be empty: {backup_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to save backup from array: {e}", exc_info=True)
        return None


def compress_tiff_file(
    file_path: str,
    output_path: Optional[str],
    compression: str,
    quality: int,
    threads: Optional[int],
    state: Optional[CompressionState] = None,
    dry_run: bool = False,
    verify_lossless_exact: bool = False,
    ignore_compression_ratio: bool = False,
    preserve_ownership: bool = False
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Compress a single TIFF file.
    
    Args:
        file_path: Path to the file to compress
        output_path: Optional output path (None for in-place)
        compression: Compression type
        quality: Compression quality
        threads: Number of threads
        state: Optional CompressionState. If None, will use state file from file's directory.
        dry_run: Whether to run in dry-run mode
        verify_lossless_exact: Whether to verify lossless compression exactly
        ignore_compression_ratio: Whether to ignore compression ratio threshold
        preserve_ownership: Whether to preserve file ownership and permissions (useful when running as root)
    
    Returns:
        (success, error_message, compression_ratio)
        compression_ratio is None if skipped or error, otherwise the ratio value
    """
    # Get state file for the directory containing the file
    if state is None:
        file_dir = os.path.dirname(file_path)
        state_file = get_state_file_for_directory(file_dir)
        state = CompressionState(state_file, preserve_ownership=preserve_ownership)
    
    # Record original file ownership and permissions if preserve_ownership is enabled
    original_ownership = None
    if preserve_ownership:
        original_ownership = get_file_ownership(file_path)
        if original_ownership is None:
            logging.warning(f"Could not get ownership for {file_path}, ownership preservation may fail")
    
    try:
        # Check file size
        original_size = os.path.getsize(file_path)
        if original_size < MIN_FILE_SIZE:
            return True, f"Skipped (file size {original_size} < {MIN_FILE_SIZE} bytes)", None

        if dry_run:
            return True, "Dry run - would compress", None

        # Check RAM size limit
        available_ram = get_available_ram()
        max_file_size = int(available_ram * RAM_SIZE_LIMIT_RATIO)
        if original_size > max_file_size:
            logging.warning(
                f"Skipping {file_path}: file size ({original_size / (1024**3):.2f} GB) "
                f"exceeds 40% of free RAM ({available_ram / (1024**3):.2f} GB, "
                f"limit: {max_file_size / (1024**3):.2f} GB)"
            )
            return True, f"Skipped (file size {original_size / (1024**2):.1f} MB > {max_file_size / (1024**2):.1f} MB, 40% of free RAM)", None

        # Verify file can be read
        can_read, array = verify_tiff_file(file_path)
        if not can_read:
            return False, "Could not read file as valid TIFF/numpy array", None


        # Determine output path
        if output_path:
            # Create output directory structure and verify write permissions
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            # Verify write permissions
            if not os.access(output_dir, os.W_OK):
                return False, f"Output directory is not writable: {output_dir}", None
            final_path = output_path
            temp_path = output_path + TEMP_SUFFIX
        else:
            final_path = file_path
            temp_path = file_path + TEMP_SUFFIX
        
        # Check disk space before compression
        # For in-place compression, need space for both original and compressed (worst case: 2x original)
        # For output compression, need space for compressed file
        if output_path:
            required_space = original_size  # Worst case: compressed is same size
        else:
            required_space = original_size * 10  # Need space for both original and compressed with some margin
        
        has_space, available = check_disk_space(
            os.path.dirname(temp_path) if output_path else os.path.dirname(file_path),
            required_space
        )
        if not has_space:
            return False, (
                f"Insufficient disk space: need {required_space / (1024**2):.1f} MB, "
                f"available {available / (1024**2):.1f} MB"
            ), None

        # Compress to temporary file
        tifffile_version = pkg_resources.get_distribution("tifffile").version
        try:
            with tifffile.TiffWriter(temp_path) as tiff:
                if tifffile_version > "2022.7.28":
                    if compression == "jpeg_2000_lossy":
                        tiff.write(
                            array,
                            compression=compression,
                            compressionargs={'level': quality},
                            maxworkers=threads
                        )
                    else:
                        tiff.write(
                            array,
                            compression=compression,
                            maxworkers=threads
                        )
                else:
                    tiff.write(
                        array,
                        compression=(compression, quality),
                        maxworkers=threads
                    )
        except Exception as e:
            rename_temp_file_on_error(temp_path, file_path, f"Compression failed: {e}")
            return False, f"Compression failed: {e}", None

        if not os.path.exists(temp_path):
            return False, "Compressed file was not created", None

        # Ensure temp file is fully written before moving (critical for HDD RAID)
        try:
            with open(temp_path, "rb") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            rename_temp_file_on_error(temp_path, file_path, f"Could not fsync temp file: {e}")
            return (
                False,
                "Compressed file verification failed - could not fsync temp file",
                None
            )

        compressed_size = os.path.getsize(temp_path)
        if compressed_size == 0:
            rename_temp_file_on_error(temp_path, file_path, "Compressed file is empty")
            return False, "Compressed file is empty", None

        # Check compression ratio
        compression_ratio = float(original_size) / compressed_size
        if not ignore_compression_ratio and compression_ratio < COMPRESSION_RATIO_THRESHOLD:
            os.remove(temp_path)
            # Mark file as skipped in state file
            state.mark_skipped(
                file_path,
                f"compression ratio {compression_ratio:.2f} < {COMPRESSION_RATIO_THRESHOLD}",
                compression_ratio
            )
            return True, f"Skipped (compression ratio {compression_ratio:.2f} < {COMPRESSION_RATIO_THRESHOLD})", None

        # Verify compressed file can be read
        can_read_compressed, compressed_array = verify_tiff_file(temp_path)
        if not can_read_compressed:
            rename_temp_file_on_error(temp_path, file_path, "Compressed file verification failed - could not read")
            return False, "Compressed file verification failed - could not read", None

        # Always compare array size and type (basic sanity check)
        if array.dtype != compressed_array.dtype:
            rename_temp_file_on_error(temp_path, file_path, f"Array dtype mismatch: original {array.dtype} vs compressed {compressed_array.dtype}")
            return False, f"Array dtype mismatch: original {array.dtype} vs compressed {compressed_array.dtype}", None
        if array.shape != compressed_array.shape:
            rename_temp_file_on_error(temp_path, file_path, f"Array shape mismatch: original {array.shape} vs compressed {compressed_array.shape}")
            return False, f"Array shape mismatch: original {array.shape} vs compressed {compressed_array.shape}", None

        # Optional: Bit-exact verification for lossless compression only
        if verify_lossless_exact:
            if compression == "jpeg_2000_lossy":
                # Skip exact verification for lossy compression
                logging.debug(f"Skipping bit-exact verification for lossy compression: {file_path}")
            else:
                # For lossless compression, verify bit-exactness
                if not arrays_equal_exact(array, compressed_array):
                    rename_temp_file_on_error(temp_path, file_path, "Compressed file verification failed - bit-exact mismatch (lossless compression)")
                    return False, "Compressed file verification failed - bit-exact mismatch (lossless compression)", None

        # Atomic move: replace original with compressed
        # Only move if temp file is verified and complete
        # For in-place compression, verify original file exists and is intact before move
        if not output_path:
            # Verify original file still exists and has correct size before attempting move
            if not os.path.exists(file_path):
                rename_temp_file_on_error(temp_path, file_path, "Original file disappeared before move")
                return False, "Original file disappeared before move - aborting to preserve data", None
            if os.path.getsize(file_path) != original_size:
                rename_temp_file_on_error(temp_path, file_path, f"Original file size changed before move (expected {original_size}, got {os.path.getsize(file_path)})")
                return False, "Original file size changed before move - aborting to preserve data", None
        
        try:
            if output_path:
                # For output folder, just move the temp file
                shutil.move(temp_path, final_path)
            else:
                # For in-place compression, use atomic replacement
                # Move temp to final location (this is atomic on local filesystem)
                shutil.move(temp_path, final_path)

                # Verify final file size matches
                final_file_size = os.path.getsize(final_path)
                if final_file_size != compressed_size:
                    # Check if move actually failed (file still has original size)
                    if final_file_size == original_size:
                        # Move failed - original file is still there, temp file still exists
                        # This is actually fine - the original is preserved
                        rename_temp_file_on_error(temp_path, file_path, "File size mismatch after move - move appears to have failed")
                        return False, "File size mismatch after move - original preserved (move failed)", None
                    else:
                        # Move succeeded but file has wrong size - this is bad
                        # Original is gone, compressed file is corrupted - save backup from array
                        backup_path = save_backup_from_array(array, file_path)
                        if backup_path:
                            return False, f"File size mismatch after move - expected {compressed_size}, got {final_file_size} (original may be corrupted, backup saved to {backup_path})", None
                        else:
                            return False, f"File size mismatch after move - expected {compressed_size}, got {final_file_size} (original may be corrupted, backup save failed)", None

        except OSError as e:
            # Verify original file is still intact after failed move
            if not output_path:
                if os.path.exists(file_path):
                    if os.path.getsize(file_path) == original_size:
                        # Original is intact, good
                        rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e}")
                        return False, f"Failed to move compressed file: {e} (original file preserved)", None
                    else:
                        # Original size changed, this is bad - save backup from array
                        backup_path = save_backup_from_array(array, file_path)
                        if backup_path:
                            rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e} - original file size changed (backup saved to {backup_path})")
                            return False, f"Failed to move compressed file: {e} - original file may be corrupted (backup saved to {backup_path})", None
                        else:
                            rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e} - original file size changed (backup save failed)")
                            return False, f"Failed to move compressed file: {e} - original file may be corrupted (backup save failed)", None
                else:
                    # Original disappeared, this is very bad - save backup from array
                    backup_path = save_backup_from_array(array, file_path)
                    if backup_path:
                        rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e} - original file disappeared (backup saved to {backup_path})")
                        return False, f"Failed to move compressed file: {e} - original file disappeared (backup saved to {backup_path})", None
                    else:
                        rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e} - original file disappeared (backup save failed)")
                        return False, f"Failed to move compressed file: {e} - original file disappeared (backup save failed)", None
            else:
                # For output path, just report the error
                rename_temp_file_on_error(temp_path, file_path, f"OSError during file move: {e}")
                return False, f"Failed to move compressed file: {e}", None

        # Restore ownership and permissions if preserve_ownership is enabled
        if preserve_ownership and original_ownership is not None:
            uid, gid, mode = original_ownership
            if not set_file_ownership(final_path, uid, gid, mode):
                logging.warning(f"Failed to restore ownership/permissions for {final_path}")
        
        state.mark_processed(
            file_path,
            compression_ratio,
            compression,
            original_size,
            compressed_size
        )

        return True, f"Compressed (ratio: {compression_ratio:.2f}x, {original_size} -> {compressed_size} bytes)", compression_ratio

    except Exception as e:
        logging.error(f"Unexpected error compressing {file_path}: {e}", exc_info=True)
        # Rename temp file if it exists
        temp_path = (output_path if output_path else file_path) + TEMP_SUFFIX
        rename_temp_file_on_error(temp_path, file_path, f"Unexpected error: {e}")
        return False, f"Unexpected error: {e}", None


def get_state_file_for_directory(directory: str) -> str:
    """Get the path to the state file for a given directory."""
    return os.path.join(directory, STATE_FILE)


def count_processed_files(root_dir: str) -> int:
    """Count total number of processed files across all directories."""
    total_processed = 0
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        state_file = get_state_file_for_directory(root)
        if os.path.exists(state_file):
            state = CompressionState(state_file)
            total_processed += state.get_processed_count()
    
    return total_processed


def count_skipped_files(root_dir: str) -> int:
    """Count total number of skipped files across all directories."""
    total_skipped = 0
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        state_file = get_state_file_for_directory(root)
        if os.path.exists(state_file):
            state = CompressionState(state_file)
            total_skipped += state.get_skipped_count()
    
    return total_skipped


def find_tiff_files(root_dir: str, force_recompress_skipped: bool = False, 
                    force_recompress_processed: bool = False) -> List[str]:
    """Find all TIFF files that need compression.
    
    Processes directories one at a time, checking for state file in each directory.
    Only files in the same directory as the state file are tracked by that state file.
    Skips hidden directories and directories named as STATE_FILE.
    
    Args:
        root_dir: Root directory to search for TIFF files
        force_recompress_skipped: If True, include files that were previously skipped
        force_recompress_processed: If True, include files that were previously processed
    """
    tiff_files = []
    processed_dirs = set()  # Track directories we've already processed

    for root, _, files in os.walk(root_dir):
        # Skip hidden directories
        if os.path.basename(root).startswith('.'):
            continue
        # Skip this directory if it is named after the STATE_FILE
        if os.path.basename(root) == STATE_FILE:
            continue

        # Skip if we've already processed this directory (shouldn't happen, but safety check)
        if root in processed_dirs:
            continue

        # Get state file for this directory
        state_file = get_state_file_for_directory(root)
        state = CompressionState(state_file)

        # Process files in this directory
        for file in files:
            # Skip state file itself
            if file == STATE_FILE:
                continue
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                # Check if already processed
                is_processed = state.is_processed(file_path)
                # Check if skipped
                is_skipped = state.is_skipped(file_path)
                
                # Include file if:
                # - Not processed and not skipped (normal case)
                # - Skipped but force_recompress_skipped is True
                # - Processed but force_recompress_processed is True
                should_include = False
                if not is_processed and not is_skipped:
                    should_include = True
                elif is_skipped and force_recompress_skipped:
                    should_include = True
                elif is_processed and force_recompress_processed:
                    should_include = True
                
                if should_include:
                    tiff_files.append(file_path)

        processed_dirs.add(root)

    return sorted(tiff_files)


def cleanup_temp_files(root_dir: str, cleanup_error_files: bool = False):
    """
    Clean up any leftover temporary files from previous runs.
    
    Args:
        root_dir: Root directory to search for temp files
        cleanup_error_files: If True, also clean up .compressing.ERROR files
    
    Note: Files with TEMP_ERROR_SUFFIX (.compressing.ERROR) are excluded from cleanup by default
    as they indicate files that encountered errors and need manual investigation.
    """
    cleaned = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Clean up regular temp files
            if file.endswith(TEMP_SUFFIX) and not file.endswith(TEMP_ERROR_SUFFIX):
                temp_path = os.path.join(root, file)
                try:
                    # Check if corresponding original exists
                    original_path = temp_path[:-len(TEMP_SUFFIX)]
                    if os.path.exists(original_path):
                        # Temp file is orphaned, remove it
                        os.remove(temp_path)
                        cleaned += 1
                        logging.info(f"Cleaned up orphaned temp file: {temp_path}")
                except Exception as e:
                    logging.warning(f"Could not clean up {temp_path}: {e}")
            
            # Optionally clean up error-marked files
            if cleanup_error_files and file.endswith(TEMP_ERROR_SUFFIX):
                error_path = os.path.join(root, file)
                try:
                    # Check if corresponding original exists
                    original_path = error_path[:-len(TEMP_ERROR_SUFFIX)]
                    if os.path.exists(original_path):
                        # Error file is orphaned, remove it
                        os.remove(error_path)
                        cleaned += 1
                        logging.info(f"Cleaned up error-marked temp file: {error_path}")
                except Exception as e:
                    logging.warning(f"Could not clean up {error_path}: {e}")
    return cleaned


def setup_logging(log_dir: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def calculate_optimal_threads() -> int:
    """Calculate optimal number of threads for tifffile compression."""
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of CPU for internal compression threads
    optimal = max(1, int(cpu_count * 0.75))
    return optimal


def process_tiff_files(
    tiff_files: List[str],
    folder: str,
    output: Optional[str],
    compression: str,
    quality: int,
    threads: int,
    state: Optional[CompressionState] = None,
    dry_run: bool = False,
    verify_lossless_exact: bool = False,
    ignore_compression_ratio: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[str, int, int, Dict], bool]] = None,
    preserve_ownership: bool = False
) -> Dict[str, int]:
    """
    Process a list of TIFF files for compression.
    
    Args:
        tiff_files: List of file paths to process
        folder: Root folder for relative path calculation
        output: Output folder (None for in-place compression)
        compression: Compression type
        quality: Compression quality
        threads: Number of threads
        state: Optional CompressionState instance. If None, will use per-directory state files.
        dry_run: Whether to run in dry-run mode
        verify_lossless_exact: Whether to verify lossless compression exactly
        ignore_compression_ratio: Whether to ignore compression ratio threshold
        show_progress: Whether to show progress bar
        progress_callback: Optional callback function(current_file, current_index, total_files, stats) -> bool
                          Called before processing each file. Returns True to continue, False to stop.
        preserve_ownership: Whether to preserve file ownership and permissions (useful when running as root)
    
    Returns:
        Dictionary with keys: success_count, skip_count, error_count, consecutive_errors, compression_ratios
        compression_ratios is a list of compression ratios for successfully compressed files
    """
    success_count = 0
    skip_count = 0
    error_count = 0
    consecutive_errors = 0
    last_errors = []  # Track last errors for warning file
    stop_processing = False
    compression_ratios = []  # Track compression ratios for successfully compressed files
    
    total_files = len(tiff_files)
    
    if show_progress:
        pbar = tqdm(total=total_files, desc="Compressing", unit="file")
    else:
        pbar = None
    
    try:
        for index, file_path in enumerate(tiff_files):
            if stop_processing:
                break
            
            # Call progress callback if provided
            if progress_callback is not None:
                stats = {
                    'success_count': success_count,
                    'skip_count': skip_count,
                    'error_count': error_count,
                    'consecutive_errors': consecutive_errors,
                    'total_files': total_files
                }
                should_continue = progress_callback(file_path, index, total_files, stats)
                if not should_continue:
                    stop_processing = True
                    logging.info("Compression stopped by progress callback")
                    break
            
            # Determine output path if output folder is specified
            output_path = None
            if output:
                rel_path = os.path.relpath(file_path, folder)
                output_path = os.path.join(output, rel_path)
            
            try:
                # Pass None for state - compress_tiff_file will get state from file's directory
                success, message, compression_ratio = compress_tiff_file(
                    file_path,
                    output_path,
                    compression,
                    quality,
                    threads,
                    None,  # Will use per-directory state file
                    dry_run,
                    verify_lossless_exact,
                    ignore_compression_ratio,
                    preserve_ownership
                )
                
                if success:
                    # Reset consecutive error counter on success
                    consecutive_errors = 0
                    last_errors = []  # Clear error history on success
                    if "Skipped" in message:
                        skip_count += 1
                    else:
                        success_count += 1
                        # Collect compression ratio if available
                        if compression_ratio is not None:
                            compression_ratios.append(compression_ratio)
                else:
                    # Track consecutive errors
                    error_count += 1
                    consecutive_errors += 1
                    last_errors.append((file_path, message))
                    # Keep only last MAX_CONSECUTIVE_ERRORS errors
                    if len(last_errors) > MAX_CONSECUTIVE_ERRORS:
                        last_errors.pop(0)
                    logging.warning(f"{file_path}: {message}")
                    
                    # Stop if we've hit max consecutive errors
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        stop_processing = True
                        logging.error(
                            f"Stopping compression after {consecutive_errors} consecutive errors. "
                            f"This indicates a serious issue that requires investigation."
                        )
                        create_warning_file(folder, consecutive_errors, last_errors)
                        break
            
            except Exception as e:
                # Track consecutive errors for exceptions too
                error_count += 1
                consecutive_errors += 1
                error_msg = f"Unexpected error: {e}"
                last_errors.append((file_path, error_msg))
                if len(last_errors) > MAX_CONSECUTIVE_ERRORS:
                    last_errors.pop(0)
                logging.error(f"{file_path}: {error_msg}", exc_info=True)
                
                # Stop if we've hit max consecutive errors
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    stop_processing = True
                    logging.error(
                        f"Stopping compression after {consecutive_errors} consecutive errors. "
                        f"This indicates a serious issue that requires investigation."
                    )
                    create_warning_file(folder, consecutive_errors, last_errors)
                    break
            
            # Update progress bar
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'OK': success_count,
                    'Skip': skip_count,
                    'Error': error_count,
                    'ConsecErr': consecutive_errors
                })
    
    finally:
        if pbar:
            pbar.close()
    
    return {
        'success_count': success_count,
        'skip_count': skip_count,
        'error_count': error_count,
        'consecutive_errors': consecutive_errors,
        'stop_processing': stop_processing,
        'compression_ratios': compression_ratios
    }


def print_compression_ratio_histogram(compression_ratios: List[float]) -> None:
    """
    Print and log a histogram of compression ratios.
    
    Args:
        compression_ratios: List of compression ratios for successfully compressed files
    """
    if not compression_ratios:
        logging.info("No compression ratios to display (no files were successfully compressed)")
        return
    
    # Calculate statistics
    ratios_array = np.array(compression_ratios)
    min_ratio = float(np.min(ratios_array))
    max_ratio = float(np.max(ratios_array))
    mean_ratio = float(np.mean(ratios_array))
    median_ratio = float(np.median(ratios_array))
    std_ratio = float(np.std(ratios_array))
    
    # Determine histogram bins
    # Use adaptive binning based on the range
    num_bins = 20
    if max_ratio - min_ratio < 0.5:
        # Very narrow range, use smaller bins
        num_bins = 10
    elif max_ratio - min_ratio > 15:
        # Very wide range, use larger bins
        num_bins = 30
    
    # Create histogram
    counts, bin_edges = np.histogram(compression_ratios, bins=num_bins)
    
    # Find max count for scaling
    max_count = int(np.max(counts))
    bar_width = 50  # Maximum width of histogram bars in characters
    
    
    # Log the same information
    logging.info("=" * 70)
    logging.info("Compression Ratio Statistics")
    logging.info("=" * 70)
    logging.info(f"  Total files compressed: {len(compression_ratios)}")
    logging.info(f"  Minimum ratio:         {min_ratio:.2f}x")
    logging.info(f"  Maximum ratio:         {max_ratio:.2f}x")
    logging.info(f"  Mean ratio:            {mean_ratio:.2f}x")
    logging.info(f"  Median ratio:          {median_ratio:.2f}x")
    logging.info(f"  Std deviation:         {std_ratio:.2f}x")
    logging.info("\nCompression Ratio Distribution:")
    logging.info("-" * 70)
    
    for i in range(len(counts)):
        count = int(counts[i])
        if count == 0:
            continue
        
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
        
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        
        logging.info(f"  {bin_label:>15} │{bar:<{bar_width}} {count:>5} files")
    
    logging.info("-" * 70)
    logging.info("=" * 70)


def create_warning_file(root_dir: str, error_count: int, last_errors: List[Tuple[str, str]]) -> None:
    """
    Create a warning file in the root directory when compression stops due to errors.
    
    Args:
        root_dir: Root directory where compression was running
        error_count: Number of consecutive errors encountered
        last_errors: List of (file_path, error_message) tuples for the last errors
    """
    warning_path = os.path.join(root_dir, WARNING_FILE)
    try:
        with open(warning_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("COMPRESSION STOPPED DUE TO CONSECUTIVE ERRORS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Compression was stopped after {error_count} consecutive errors.\n")
            f.write(f"This indicates a serious issue that requires investigation.\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write("Last errors encountered:\n")
            f.write("-" * 60 + "\n")
            for i, (file_path, error_msg) in enumerate(last_errors, 1):
                f.write(f"\nError {i}:\n")
                f.write(f"  File: {file_path}\n")
                f.write(f"  Error: {error_msg}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Please investigate the errors and temporary files marked with .compressing.ERROR\n")
            f.write("before resuming compression.\n")
        logging.error(f"Created warning file: {warning_path}")
    except Exception as e:
        logging.error(f"Failed to create warning file {warning_path}: {e}")


def run_compression(
    folder: str,
    compression: str = 'zlib',
    quality: int = 85,
    threads: Optional[int] = None,
    output: Optional[str] = None,
    dry_run: bool = False,
    cleanup_temp: bool = False,
    cleanup_error_files: bool = False,
    verify_lossless_exact: bool = False,
    ignore_compression_ratio: bool = False,
    force_recompress_skipped: bool = False,
    force_recompress_processed: bool = False,
    progress_callback: Optional[Callable[[str, int, int, Dict], bool]] = None,
    preserve_ownership: bool = False,
    save_log_file: bool = True
) -> Dict:
    """
    Run compression on TIFF files in the specified folder.
    
    Args:
        folder: Folder containing TIFF files to compress (recursive)
        compression: Compression algorithm ('zlib' or 'jpeg_2000_lossy')
        quality: Compression quality for lossy compression (0-100)
        threads: Number of threads for tifffile compression (None for auto)
        output: Output folder (None for in-place compression)
        dry_run: Show what would be compressed without actually compressing
        cleanup_temp: Clean up temporary files from previous runs before starting
        cleanup_error_files: Also clean up .compressing.ERROR files
        verify_lossless_exact: Verify bit-exact match for lossless compression
        ignore_compression_ratio: Ignore compression ratio threshold requirement
        force_recompress_skipped: Force recompression of previously skipped files
        force_recompress_processed: Force recompression of previously processed files
        progress_callback: Optional callback function(current_file, current_index, total_files, stats) -> bool
                          Returns True to continue, False to stop compression
        preserve_ownership: Whether to preserve file ownership and permissions (useful when running as root in Docker)
                           When enabled, compressed files retain original ownership/permissions, and STATE_FILE
                           will have ownership matching its directory with read+write for owner.
    
    Returns:
        Dictionary with compression results including success_count, skip_count, error_count, etc.
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If compression fails
    """
    # Validate inputs
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a directory")
    
    if output and not os.path.isdir(os.path.dirname(os.path.abspath(output))):
        raise ValueError(f"Parent directory of {output} does not exist")
    
    if quality < 0 or quality > 100:
        raise ValueError("Quality must be between 0 and 100")
    
    # Setup logging
    logging_dir = None
    if save_log_file:
        logging_dir = folder
    setup_logging(log_dir=logging_dir)
    
    # Calculate threads for tifffile compression
    if threads is None:
        threads = calculate_optimal_threads()
    
    # Get available RAM info (fail if detection fails)
    try:
        available_ram = get_available_ram()
    except RuntimeError as e:
        logging.error(f"Failed to detect free RAM: {e}")
        raise RuntimeError(f"Failed to detect free RAM: {e}") from e
    max_file_size = int(available_ram * RAM_SIZE_LIMIT_RATIO)
    
    logging.info("Starting compression (sequential processing)")
    logging.info(f"Tifffile compression threads: {threads}")
    logging.info(f"Compression: {compression}, Quality: {quality}")
    logging.info(f"Free RAM: {available_ram / (1024**3):.2f} GB, Max file size: {max_file_size / (1024**3):.2f} GB (40% of free RAM)")
    logging.info(f"Folder: {folder}")
    if output:
        logging.info(f"Output folder: {output}")
    if dry_run:
        logging.info("DRY RUN MODE - No files will be modified")
    if verify_lossless_exact:
        logging.info("BIT-EXACT VERIFICATION MODE - Lossless compressed files will be verified for bit-exact match")
    if force_recompress_skipped:
        logging.info("FORCE RECOMPRESS SKIPPED MODE - Will recompress all previously skipped files")
    if force_recompress_processed:
        logging.info("FORCE RECOMPRESS PROCESSED MODE - Will recompress all previously processed files")
    if preserve_ownership:
        logging.info("PRESERVE OWNERSHIP MODE - File ownership and permissions will be preserved")
    
    # Disclaimer about local filesystem requirement
    logging.info("NOTE: This tool is designed for LOCAL filesystems only. Network filesystems are not supported.")
    logging.info("NOTE: State files are created per directory. Each directory tracks only its own files.")
    
    # Cleanup temp files if requested
    if cleanup_temp:
        cleaned = cleanup_temp_files(folder, cleanup_error_files=cleanup_error_files)
        if cleaned > 0:
            logging.info(f"Cleaned up {cleaned} temporary files")
        if cleanup_error_files:
            logging.warning("Cleaned up error-marked files - ensure you've investigated any issues")
    
    # Acquire lock
    lock_file = os.path.join(folder, LOCK_FILE)
    with FileLock(lock_file):
        # Find files to compress
        logging.info("Scanning for TIFF files...")
        tiff_files = find_tiff_files(
            folder,
            force_recompress_skipped=force_recompress_skipped,
            force_recompress_processed=force_recompress_processed
        )
        total_files = len(tiff_files)
        processed_count = count_processed_files(folder)
        skipped_count = count_skipped_files(folder)
        
        logging.info(f"Found {total_files} files to compress ({processed_count} already processed, {skipped_count} already skipped across all directories)")
        
        if total_files == 0:
            logging.info("No files to compress")
            return {
                'success_count': 0,
                'skip_count': 0,
                'error_count': 0,
                'consecutive_errors': 0,
                'stop_processing': False,
                'compression_ratios': []
            }
        
        # Process files sequentially
        results = process_tiff_files(
            tiff_files,
            folder,
            output,
            compression,
            quality,
            threads,
            None,  # Will use per-directory state files
            dry_run,
            verify_lossless_exact,
            ignore_compression_ratio,
            show_progress=True,
            progress_callback=progress_callback,
            preserve_ownership=preserve_ownership
        )
        
        # Summary
        logging.info("=" * 60)
        logging.info("Compression Summary:")
        logging.info(f"  Successfully compressed: {results['success_count']}")
        logging.info(f"  Skipped: {results['skip_count']}")
        logging.info(f"  Errors: {results['error_count']}")
        logging.info(f"  Already processed (across all directories): {processed_count}")
        logging.info(f"  Already skipped (across all directories): {skipped_count}")
        if results['stop_processing']:
            logging.warning(f"  Compression stopped early after {results['consecutive_errors']} consecutive errors")
            logging.warning(f"  Warning file created: {os.path.join(folder, WARNING_FILE)}")
        logging.info("=" * 60)
        
        # Print compression ratio histogram
        print_compression_ratio_histogram(results.get('compression_ratios', []))
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Robust TIFF compression tool for TrueNAS Scale storage servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress in place with zlib compression
  %(prog)s --folder /mnt/data/images --compression zlib

  # Compress to output folder with JPEG2000
  %(prog)s --folder /mnt/data/images --compression jpeg_2000_lossy --quality 85 --output /mnt/data/compressed

  # Dry run to see what would be compressed
  %(prog)s --folder /mnt/data/images --compression zlib --dry-run
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Folder containing TIFF files to compress (recursive)'
    )
    
    parser.add_argument(
        '--compression',
        type=str,
        choices=['zlib', 'jpeg_2000_lossy'],
        default='zlib',
        help='Compression algorithm (default: zlib)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        help='Compression quality for lossy compression (0-100, default: 85)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Number of threads for tifffile compression (default: 75%% of CPU cores)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output folder (if specified, files are not compressed in place)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be compressed without actually compressing'
    )
    
    parser.add_argument(
        '--cleanup-temp',
        action='store_true',
        help='Clean up temporary files from previous runs before starting'
    )
    
    parser.add_argument(
        '--cleanup-error-files',
        action='store_true',
        help='Also clean up .compressing.ERROR files (use with caution - these indicate files that need investigation)'
    )
    
    parser.add_argument(
        '--verify-lossless-exact',
        action='store_true',
        help='For lossless compression (zlib), verify bit-exact match by comparing arrays. '
             'For lossy compression (jpeg_2000_lossy), this option is ignored. '
             'Slower but ensures data integrity for lossless compression.'
    )
    
    parser.add_argument(
        '--ignore-compression-ratio',
        action='store_true',
        help='Ignore compression ratio threshold requirement. '
             'Useful for testing or when you want to compress files regardless of ratio.'
    )
    
    parser.add_argument(
        '--force-recompress-skipped',
        action='store_true',
        help='Force recompression of all previously skipped files that are recorded in STATE_FILE. '
             'Files are skipped when compression ratio is too small or other skip conditions are met.'
    )
    
    parser.add_argument(
        '--force-recompress-processed',
        action='store_true',
        help='Force recompression of all previously processed files that are recorded in STATE_FILE. '
             'This will recompress files that were successfully compressed in previous runs.'
    )
    
    parser.add_argument(
        '--preserve-ownership',
        action='store_true',
        help='Preserve file ownership and permissions when compressing files. '
             'Useful when running as root (e.g., in Docker containers). '
             'Compressed files will retain original ownership/permissions, '
             'and STATE_FILE will have ownership matching its directory with read+write for owner.'
             'Also set by PRESERVE_OWNERSHIP environment variable.'
    )
    
    args = parser.parse_args()
    
    # Check environment variable as fallback (useful for Docker)
    preserve_ownership = args.preserve_ownership or os.getenv('PRESERVE_OWNERSHIP', '').lower() in ('1', 'true', 'yes')
    
    try:
        run_compression(
            folder=args.folder,
            compression=args.compression,
            quality=args.quality,
            threads=args.threads,
            output=args.output,
            dry_run=args.dry_run,
            cleanup_temp=args.cleanup_temp,
            cleanup_error_files=args.cleanup_error_files,
            verify_lossless_exact=args.verify_lossless_exact,
            ignore_compression_ratio=args.ignore_compression_ratio,
            force_recompress_skipped=args.force_recompress_skipped,
            force_recompress_processed=args.force_recompress_processed,
            preserve_ownership=preserve_ownership
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        logging.error(str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
