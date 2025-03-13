# Contributing Guidelines

Thank you for your interest in contributing to the Regulatory Compliance Analysis Tool! This document provides guidelines and instructions for contributing.

## Code of Conduct

We expect all contributors to adhere to our Code of Conduct. Please be respectful of others and create a positive environment for collaboration.

## Getting Started

### Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/improved-reg-review.git
   cd improved-reg-review
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Development Workflow

### Branching Strategy

We follow the GitHub Flow strategy:

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Commit frequently with clear messages:
   ```bash
   git commit -m "Add feature: description of change"
   ```
4. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a Pull Request against `main`

### Coding Standards

We follow PEP 8 and use black for code formatting:

```bash
# Format code
black compliance_poc/

# Check for linting issues
flake8 compliance_poc/
```

### Testing

All new features should include tests:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=compliance_poc
```

## Pull Request Process

1. Update documentation for any changed functionality
2. Add or update tests as necessary
3. Ensure all tests pass and code quality checks succeed
4. If you've added new dependencies, update requirements.txt
5. Submit your PR with a comprehensive description of changes

## Types of Contributions

### Bug Fixes

- Clearly describe the bug in the PR
- Include steps to reproduce
- Reference any related issues

### Feature Additions

- Discuss major features in an issue before starting work
- Keep features focused and modular
- Include documentation and tests

### Documentation Improvements

- Ensure accurate and clear language
- For code examples, verify they work as documented

## Style Guide

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- First line is a concise summary (50 chars or less)
- Follow with a detailed description if necessary

### Python Style

- Class names: `CamelCase`
- Function names: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods/variables: prefix with underscore (`_private_method`)

## Review Process

All submissions require review before merging:

1. Automated checks must pass (tests, linting)
2. At least one maintainer must approve changes
3. Discussions and requested changes must be resolved

## License

By contributing, you agree that your contributions will be licensed under the project's license.
