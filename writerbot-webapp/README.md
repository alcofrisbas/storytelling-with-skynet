The webapp requires the following Python packages:

```
django
python-decouple
python-social-auth[django]
```

To run, you must create a .env file in the writerbot-webapp directory with the following contents:

```
DJANGO_SECRET_KEY=<some string of numbers and letters>
GOOGLE_CLIENT_ID=<>
GOOGLE_CLIENT_SECRET=<>
```

For local testing, you can set DJANGO_SECRET_KEY to whatever you like, but the Google entries must be set to valid Google webapp credentials.

If this is your first time running the webapp and you don't have a db.sqlite3 file, run the following command:

```bash
$ python3 manage.py migrate
```

Then run the webserver with:

```bash
$ python3 manage.py runserver localhost:8000
```

You can then load localhost:8000 in a browser to view the webapp.
