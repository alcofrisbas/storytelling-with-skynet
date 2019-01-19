#!/bin/sh

rm writerbot-webapp/db.sqlite3
venv/bin/python writerbot-webapp/manage.py migrate --run-syncdb