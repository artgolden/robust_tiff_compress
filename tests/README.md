# Test Suite for robust_tiff_compress

This directory contains a comprehensive test suite for the `robust_tiff_compress` module, covering all major functionality including compression operations, state management, file locking, error handling, and integration scenarios.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and test utilities
├── test_compression.py            # Core compression functionality tests
├── test_state_management.py      # CompressionState and resumability tests
├── test_file_locking.py           # FileLock and concurrent run prevention
├── test_error_handling.py         # Failure state tests
├── test_utilities.py              # Helper function tests (RAM, disk space, etc.)
├── test_integration.py            # End-to-end integration tests
└── fixtures/
    └── sample_tiffs/              # Pre-generated test TIFF files (if needed)
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

### Basic Test Execution

Run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_compression.py
```

Run specific test class or function:

```bash
pytest tests/test_compression.py::TestBasicCompression
pytest tests/test_compression.py::TestBasicCompression::test_compress_medium_file_in_place
```

### Test Coverage

Run tests with coverage report:

```bash
pytest --cov=robust_tiff_compress --cov-report=html --cov-report=term-missing
```

This will generate:
- Terminal output with coverage summary
- HTML report in `htmlcov/index.html`

### Running Specific Test Categories

Run only unit tests (exclude integration tests):

```bash
pytest -m "not integration"
```

Run only fast tests (exclude slow tests):

```bash
pytest -m "not slow"
```

### Parallel Test Execution

Run tests in parallel (faster execution):

```bash
pytest -n auto
```

Requires `pytest-xdist` to be installed.

## Test Organization

### test_compression.py
Tests core compression functionality:
- Basic compression for both `zlib` and `jpeg_2000_lossy`
- In-place and output directory compression
- File size handling (small, medium, large files)
- Data integrity verification
- Nested directory structures
- Dry-run mode

### test_state_management.py
Tests state file operations and resumability:
- State file loading and saving
- Marking files as processed
- Resuming after interruption
- Thread safety

### test_file_locking.py
Tests file locking mechanism:
- Lock acquisition and release
- Stale lock detection and cleanup
- Concurrent run prevention
- Error handling

### test_error_handling.py
Tests error handling and failure states:
- Disk space failures
- File permission errors
- Corrupted file handling
- Compression failures
- Consecutive error handling
- Temp file cleanup

### test_utilities.py
Tests utility functions:
- RAM detection (with psutil and fallback)
- Disk space checking
- Process checking
- File verification
- Array comparison
- File finding

### test_integration.py
End-to-end integration tests:
- Complete compression workflows
- Command-line interface
- Real-world scenarios with many files
- State persistence across runs

## Fixtures

The `conftest.py` file provides many reusable fixtures:

### Directory Fixtures
- `tmp_test_dir`: Temporary test directory
- `nested_test_dir`: Nested directory structure
- `output_dir`: Output directory for compression tests

### TIFF File Fixtures
- `small_tiff_file`: File smaller than MIN_FILE_SIZE
- `medium_tiff_file`: Medium-sized file (2-5MB)
- `large_tiff_file`: Large file (10-20MB)
- `very_large_tiff_file`: Very large file (>20% RAM)
- `tiff_file_uint8`, `tiff_file_uint16`, `tiff_file_float32`: Different dtypes
- `tiff_file_3d`: 3D stack
- `tiff_file_compressible`: File that compresses well
- `tiff_file_not_compressible`: File that doesn't compress well

### Mock Fixtures
- `mock_ram_large`, `mock_ram_small`: Mock RAM detection
- `mock_disk_space_sufficient`, `mock_disk_space_insufficient`: Mock disk space
- `mock_process_running`, `mock_process_not_running`: Mock process checking

### State and Lock Fixtures
- `state_file`: Clean state file
- `existing_state_file`: State file with existing data
- `corrupted_state_file`: Corrupted state file
- `lock_file`: Lock file path
- `existing_lock_file`: Existing lock file
- `stale_lock_file`: Stale lock file

### Error Condition Fixtures
- `corrupted_tiff_file`: Invalid TIFF file
- `empty_tiff_file`: Empty file
- `truncated_tiff_file`: Truncated TIFF file
- `read_only_dir`, `read_only_file`: Permission test fixtures

## Writing New Tests

### Test Naming Convention

Follow the pattern: `test_<functionality>_<scenario>_<expected_result>`

Example:
```python
def test_compress_medium_file_in_place_success():
    """Test compressing a medium file in place successfully."""
    ...
```

### Using Fixtures

```python
def test_example(medium_tiff_file, state_file, mock_ram_large):
    """Example test using fixtures."""
    state = CompressionState(str(state_file))
    # Test code here
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple similar cases:

```python
@pytest.mark.parametrize("compression", ["zlib", "jpeg_2000_lossy"])
def test_compression_types(compression, medium_tiff_file):
    """Test both compression types."""
    ...
```

### Mocking

Use `unittest.mock` for isolating units and creating failure conditions:

```python
from unittest.mock import patch

def test_with_mock(mock_ram_large):
    with patch('robust_tiff_compress.some_function', return_value=42):
        # Test code
        ...
```

## Continuous Integration

Tests are automatically run on:
- Push to main/master/develop branches
- Pull requests
- Weekly schedule (Sundays)

The CI configuration (`.github/workflows/test.yml`) runs tests on:
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Ubuntu latest
- With coverage reporting

## Troubleshooting

### Tests Failing Due to File Permissions

Some tests modify file permissions. If tests fail with permission errors:
- Ensure tests are run in a writable directory
- Check that temporary directories can be created

### Tests Failing Due to Missing Dependencies

Install all test dependencies:
```bash
pip install -r requirements-test.txt
```

### Slow Tests

Some tests may be slow due to file I/O. Use markers to skip:
```bash
pytest -m "not slow"
```

### Memory Issues

Some tests create large files. If you encounter memory issues:
- Reduce the number of parallel test workers: `pytest -n 2`
- Run tests individually
- Check available disk space

## Coverage Goals

The test suite aims for:
- >90% code coverage
- All error paths tested
- All compression types tested
- All helper functions tested
- All edge cases identified in code

Check current coverage:
```bash
pytest --cov=robust_tiff_compress --cov-report=term-missing
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Cleanup**: Use fixtures that automatically clean up (like `tmp_path`)
3. **Descriptive Names**: Test names should clearly describe what they test
4. **Documentation**: Add docstrings to test functions
5. **Assertions**: Use clear assertion messages
6. **Mocking**: Mock external dependencies and system calls
7. **Edge Cases**: Test both success and failure paths

## Contributing

When adding new functionality to `robust_tiff_compress.py`:

1. Add corresponding tests in the appropriate test file
2. Add new fixtures to `conftest.py` if needed
3. Ensure tests pass: `pytest tests/`
4. Check coverage: `pytest --cov=robust_tiff_compress`
5. Update this README if test structure changes significantly

