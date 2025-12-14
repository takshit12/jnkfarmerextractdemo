#!/bin/bash
echo "Starting AgriStack Web UI..."
echo

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address localhost
