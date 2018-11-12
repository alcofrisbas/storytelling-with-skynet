# webscrape-gutenberg

A simple webscraper that preprocesses data for use in our RNN

### Usage

```

usage: scraper.py [-h] [--makedb] [--all] [-q QUERY QUERY]
                  [--output_dir OUTPUT_DIR] [--test]
                  [--range SCAN_RANGE SCAN_RANGE] [--count]
                  [--random-count RANDOM_COUNT]
                  fname

positional arguments:
  fname                 specify database file

optional arguments:
  -h, --help            show this help message and exit
  --makedb              intitialize new database
  --all, -a             not yet implemented
  -q QUERY QUERY        An sql query to return records and download documents
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        destination to store documents
  --test
  --range SCAN_RANGE SCAN_RANGE, -r SCAN_RANGE SCAN_RANGE
  --count
  --random-count RANDOM_COUNT

```

To initialize the database:

``` python3 scraper.py fname.db --makedb -r 0 [n]``` 

where n is the number of books to query

To query for books to access:

```python3 scraper.py fname.db -q author "carroll" [--output-dir books_to_save]```

To save ```n``` random books:

```python scraper.py fname.db --random-count n [--output-dir books_to_save]```
