#!/bin/bash

# Run parameter search script for Agentic RAG optimization
# This script will test different parameter combinations to find optimal settings

echo "=========================================="
echo "Starting Parameter Search for Agentic RAG"
echo "=========================================="
echo ""
echo "This will test the following parameters:"
echo "1. Max Reflection Iterations: 2, 3, 4, 5, 6, 7"
echo "2. Temperature: 0.1, 0.25, 0.5, 0.75, 1.0"
echo "3. Number of Retrieved Docs: 1, 2, 3, 4, 5"
echo ""
echo "Results will be saved to: paper/data/parameter_search/"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Create output directory if it doesn't exist
mkdir -p paper/data/parameter_search

# Run the parameter search
uv run python paper/experiments/parameter_search.py

echo ""
echo "=========================================="
echo "Parameter Search Complete!"
echo "Check paper/data/parameter_search/ for results"
echo "==========================================">
