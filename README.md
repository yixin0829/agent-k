# Agent-K: AI-Powered Knowledge Base Assistant

Agent-K is a powerful knowledge base agentic system that uses LLM agents to help you interact with tables and reports.

# DB Agent
## Features

- 🤖 OpenAI GPT-4 powered SQL assistance
- 🔧 Automatic SQL error correction
- 📊 Schema introspection and validation
- 💾 Persistent data storage through DuckDB
- 📝 CSV export support

## Prerequisites

- Python 3.12+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-k.git
cd agent-k
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your settings:

## Data Setup

The project includes a comprehensive data setup process that handles all necessary data downloads and initialization. Run the following command to execute all setup steps:

```bash
python -m agent_k.setup.setup_all
```

This will execute the following steps in sequence:

1. **MRDS Data**: Downloads and filters Mineral Resources Data System (MRDS) data for the configured commodity
2. **DuckDB Integration**: Loads the filtered MRDS data into DuckDB for efficient querying
3. **MinMod Hyper Data**: Downloads and enriches MinMod Hyper data (ground truth dataset)
4. **43-101 Reports**: Downloads 43-101 mineral reports in PDF format concurrently
5. **Evaluation Dataset**: Constructs an match-based question eval dataset in JSONL format

The setup process is configurable through environment variables. Ensure your `.env` file includes the necessary commodity settings before running the setup.

## Using the Database Agent

Here's a simple example of using the database agent:

```bash
python -m agent_k.agents.db_agent
```

## Development

- Code style: We use `ruff` for linting
- Type checking: We use `mypy` for type checking
- Pre-commit hooks: Run `pre-commit install` to set up and `pre-commit run --all-files` to run all hooks before committing
- Python version: Make sure to use Python 3.12+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
