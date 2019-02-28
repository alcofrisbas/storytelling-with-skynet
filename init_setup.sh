#!/bin/bash
python3 -m virtualenv venv
./venv/bin/pip3 install django python-decouple python-social-auth[django] numpy tensorflow nltk gensim django-tz-detect
