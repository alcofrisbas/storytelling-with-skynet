To run, you must create a .env file in the writerbot-webapp directory with the following contents:

```
DJANGO_SECRET_KEY=<some string of numbers and letters>
```

For local testing, you can set it to whatever you like.

Then run the webserver with:

```bash
$ python3 manage.py runserver localhost:8000
```

You can then load localhost:8000 in a browser to view the webapp.
