#!/usr/bin/env bash
set -euo pipefail

python3 -m venv webshop_venv
source webshop_venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_webshop_monitoring.txt

echo
echo "WebShop monitoring environment created."
echo
echo "Next steps:"
echo "  1. source webshop_venv/bin/activate"
echo "  2. cd ~/vllmPatrickMonitoring/WebShop"
echo "  3. pip install -e ."
echo
echo "Also follow any extra official WebShop installation steps from the WebShop README."
echo "This script does not modify system Python."
