#!/bin/sh

# run this when you have changed the structure/
# added models to our code

rm writerbot-webapp/db.sqlite3
venv/bin/python writerbot-webapp/manage.py migrate --run-syncdb
