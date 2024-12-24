#!/bin/bash

CONTAINER_NAME="agent_k_postgres"

# Check if the container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} does not exist."
    exit 0
fi

# Stop the container if it's running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping PostgreSQL container..."
    docker stop ${CONTAINER_NAME}
fi

# Ask if user wants to remove the container and volume
read -p "Do you want to remove the container and its data? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing container..."
    docker rm ${CONTAINER_NAME}

    echo "Removing volume..."
    docker volume rm ${CONTAINER_NAME}_data

    echo "PostgreSQL container and data removed."
else
    echo "PostgreSQL container stopped. Data is preserved."
fi
