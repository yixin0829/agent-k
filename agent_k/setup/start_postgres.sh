#!/bin/bash

# Read .env file
source .env

# Default values from .env file
POSTGRES_DB=${POSTGRES_DB:-${DB_NAME}}
POSTGRES_USER=${POSTGRES_USER:-${DB_USER}}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-${DB_PASSWORD}}
POSTGRES_HOST=${POSTGRES_HOST:-${DB_HOST}}
POSTGRES_PORT=${POSTGRES_PORT:-${DB_PORT}}
CONTAINER_NAME="agent_k_postgres"

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Function to check if the container already exists
check_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container ${CONTAINER_NAME} already exists."
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container is already running."
            exit 0
        else
            echo "Starting existing container..."
            docker start ${CONTAINER_NAME}
            exit 0
        fi
    fi
}

# Main execution
echo "Starting PostgreSQL container..."

# Check prerequisites
check_docker
check_container

# Create a Docker volume for persistent data
docker volume create ${CONTAINER_NAME}_data

# Start PostgreSQL container
docker run -d \
    --name ${CONTAINER_NAME} \
    -e POSTGRES_DB=${POSTGRES_DB} \
    -e POSTGRES_USER=${POSTGRES_USER} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
    -v ${CONTAINER_NAME}_data:/var/lib/postgresql/data \
    -p ${POSTGRES_PORT}:5432 \
    postgres:16-alpine

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec ${CONTAINER_NAME} pg_isready -U ${POSTGRES_USER} > /dev/null 2>&1; then
        echo "PostgreSQL is ready!"
        echo "Connection details:"
        echo "  Host: ${POSTGRES_HOST}"
        echo "  Port: ${POSTGRES_PORT}"
        echo "  Database: ${POSTGRES_DB}"
        echo "  User: ${POSTGRES_USER}"
        echo "  Password: ${POSTGRES_PASSWORD}"
        exit 0
    fi
    sleep 1
done

echo "Error: PostgreSQL failed to start within 30 seconds."
exit 1
