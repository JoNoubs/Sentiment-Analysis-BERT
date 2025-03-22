#!/bin/bash
# setup.sh
# Purpose: Automate project setup
python3 -m venv sentiment_env
source sentiment_env/bin/activate
pip install -r requirements.txt
echo "Setup complete. Activate with 'source sentiment_env/bin/activate'"



