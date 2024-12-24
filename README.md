# Agent-K: AI-Powered Knowledge Base Assistant

Agent-K is a powerful knowledge base agentic system that uses LLM agents to help you interact with tables and reports.

# DB Agent
## Features

- 🤖 OpenAI GPT-4 powered SQL assistance
- 🔧 Automatic SQL error correction
- 📊 Schema introspection and validation
- 🐳 Easy PostgreSQL setup with Docker
- 💾 Persistent data storage
- 📝 CSV export support

## Prerequisites

- Python 3.12+
- Docker
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
```env
# PostgreSQL connection settings
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

## Starting PostgreSQL

1. Start the PostgreSQL container:
```bash
./agent_k/setup/start_postgres.sh
```

This will:
- Create a Docker container named `agent_k_postgres`
- Set up persistent storage
- Configure PostgreSQL with your settings
- Wait for the server to be ready

2. To stop PostgreSQL:
```bash
./agent_k/setup/stop_postgres.sh
```

## Using the Database Agent

Here's a simple example of using the database agent:

```python
from agent_k.agents.db_agent import DatabaseAgent

# Initialize the agent
agent = DatabaseAgent(
    model="gpt-4-1106-preview",  # OpenAI model to use
    temperature=0,               # Lower values = more deterministic
    max_retries=3,              # Number of retry attempts for failed queries
    output_dir="data/query_results"  # Where to save query results
)

# Connect to PostgreSQL
agent.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

# Execute a query
result = agent.execute("SELECT * FROM customers")
print(f"Success: {result['success']}")
print(f"Message: {result['message']}")
print(f"Data: {result['data']}")
```

### Error Correction Example

The agent can automatically fix SQL errors:

```python
# This query has errors
wrong_query = """
SELECT customer_name, SUM(total_amount)  -- wrong column name
FROM customers
JOIN orders ON customers.id = orders.customer_id  -- wrong column name
GROUP BY customer_name
"""

result = agent.execute(wrong_query)
print(f"Corrected query: {result.get('corrected_query')}")
```

### Running the Demo

A complete demo script is included:

```bash
python -m agent_k.examples.db_agent_demo
```

This will:
1. Set up sample tables (customers, orders, order_items)
2. Insert test data
3. Run example queries
4. Demonstrate error correction

## Development

- Code style: We use `ruff` for linting
- Pre-commit hooks: Run `pre-commit install` to set up
- Python version: Make sure to use Python 3.12+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
