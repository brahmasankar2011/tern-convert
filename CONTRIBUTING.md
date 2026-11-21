# Contributing to tern-convert

Thank you for your interest in contributing to tern-convert! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
git clone https://github.com/brahmasankar2011/tern-convert.git
cd tern-convert
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -q
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for all public functions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass: `pytest -q`
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## Sign-off

All commits must be signed off using `-s`:

```bash
git commit -s -m "Your commit message"
```

This indicates your consent to the Developer Certificate of Origin (DCO).

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.