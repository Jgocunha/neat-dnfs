#!/usr/bin/env bash
set -e

echo "Starting NEAT-DNFS dashboard..."
echo "-------------------------------"

# Change directory to the folder containing neat-dnfs-visualizer.py
cd "$(dirname "$0")"

# Launch Streamlit using Python module syntax
python3 -m streamlit run neat-dnfs-visualizer.py

echo
echo "Streamlit has stopped."
