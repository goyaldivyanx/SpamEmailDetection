#!/usr/bin/env bash
set -e

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create a directory under your project for nltk data
mkdir -p ./nltk_data

# Download only the stopwords corpus into that folder
python -m nltk.downloader stopwords -d ./nltk_data