# Contributing to AdaptGauge Core

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/ShuntaroOkuma/adapt-gauge-core.git
cd adapt-gauge-core
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

1. Fork the repository and create a feature branch
2. Make your changes
3. Run tests: `python -m pytest tests/ -v`
4. Ensure all tests pass before submitting a PR

## What We Welcome

- Bug fixes with test cases
- New model client integrations (see `src/adapt_gauge_core/infrastructure/model_clients/`)
- New scoring methods (see `src/adapt_gauge_core/scoring/`)
- Task pack contributions (see `tasks/`)
- Documentation improvements

## Code Style

- Python 3.11+
- Type hints are encouraged
- Keep functions small and focused
- Follow existing patterns in the codebase

## Reporting Issues

Please open a GitHub Issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
