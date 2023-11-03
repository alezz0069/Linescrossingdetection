#!/bin/bash

# Update system packages
sudo apt-get update

# Install poppler-utils for pdf2image
sudo apt-get install -y poppler-utils

# Install Python dependencies
pip install -r requirements.txt
