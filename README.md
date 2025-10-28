# Python Project

A modern Python project template with best practices.

## Features

- Clean project structure
- Dependency management with pip
- Testing setup with pytest
- Code formatting with black
- Linting with flake8
- Type checking with mypy

## Project Structure

```
thesis_experiments/
├── src/
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── Makefile
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis_experiments
```

2. Create a virtual environment:
```bash
make setup-venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
make install-dev
```

## Usage

Run the main application:
```bash
make run
```

Or directly:
```bash
python src/main.py
```

The application will:
- Send matrix multiplication requests every 2 seconds
- Use large matrices (1000×200 and 200×150)
- Send requests to `http://localhost:8080/submit_task`
- Continue until you press Ctrl+C

Run tests:
```bash
make test
```

Format code:
```bash
make format
```

Lint code:
```bash
make lint
```

Type check:
```bash
make lint
```

## Development

This project uses:
- **pytest** for testing
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

## License

MIT License
