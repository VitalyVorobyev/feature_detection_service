# Feature Detection Service Tests

This directory contains tests for the Feature Detection Service (FDS) application using pytest.

## Test Structure

The test suite is organized as follows:

- `test_fds.py` - Basic unit tests for various functions in the FDS application
- `test_load_image.py` - Focused tests for the `load_image_from_iss` function
- `test_integration.py` - Integration tests that verify the complete feature detection workflow
- `conftest.py` - Common test fixtures

## Mocked Dependencies

The tests mock the following external dependencies:

1. **Image Storage Service (ISS)**
   - The `requests.get` call to fetch images from ISS is mocked
   - The `requests.post` call to store feature artifacts in ISS is mocked

## Running Tests

Install the required packages:

```bash
pip install -r requirements.txt
```

Run all tests:

```bash
python -m pytest
```

Run with verbose output:

```bash
python -m pytest -v
```

Run a specific test file:

```bash
python -m pytest test_load_image.py
```

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_image` - Creates a test image with features that can be detected
- `mock_iss_response` - Mock response for ISS service API calls
- `mock_artifact_response` - Mock response for artifact creation in ISS

## Key Test Cases

1. **Unit Tests**
   - Testing individual functions like `make_detector`, `params_hash`, etc.
   - Testing the `load_image_from_iss` function with various mocked responses

2. **Integration Tests**
   - End-to-end feature detection with different algorithms
   - Handling cases where no features are detected
   - Testing with custom parameters for each algorithm
