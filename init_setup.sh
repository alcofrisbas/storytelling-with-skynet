#!/bin/bash
python3 -m virtualenv venv
./venv/bin/pip3 install django python-decouple[django] numpy tensorflow nltk
