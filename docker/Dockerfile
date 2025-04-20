FROM python:3.12

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser

# Check if requirements.txt exists and install dependencies if it does
COPY --chown=sandboxuser:sandboxuser requirements.txt* /home/sandboxuser/ 2>/dev/null || echo "No requirements.txt found, skipping pip install"
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir --user -r requirements.txt; \
    else \
        echo "No requirements.txt found in project root. Running with base Python 3.12 installation."; \
    fi


CMD ["python", "--version"]
