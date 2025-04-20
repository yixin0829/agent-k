FROM python:3.12

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser

# The wildcard is used to copy the requirements.txt file if it exists
COPY requirements.txt* .
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir --user -r requirements.txt; \
    else \
        echo "No requirements.txt found in project root. Running with base Python 3.12 installation."; \
    fi


CMD ["python", "--version"]
