#!/bin/bash
# Quick test of the ASL translation system

source asl_env/bin/activate
cd src
echo "Testing ASL translation..."
python inference.py --input "YESTERDAY I GO STORE WITH FRIEND"
python inference.py --input "MORNING COFFEE I DRINK HOT"
echo "✅ Quick test completed!"
