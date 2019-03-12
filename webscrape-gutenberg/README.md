# webscrape-gutenberg

A simple webscraper that preprocesses data for use in our RNN

### Usage

```

usage: webscrape.py [-h] command

positional arguments:
  command     Subcommand to run
              make: makes a database
              query: queries an existing database
              random: returns random books from a database.

              For more detailed help, run <command> -h.

optional arguments:
  -h, --help  show this help message and exit
```

To initialize the database:

``` python3 webscrape-gutenberg/webscrape.py make name.db -r 0 580000``` 

To query for books to access:

```python3 webscrape-gutenberg/webscrape.py query fname.db author "carroll" [--output-dir books_to_save]```

