import argparse
from argparse import RawTextHelpFormatter
import sys
import os
import scraper as scp


class ScrapeArgs:

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
        parser.add_argument('command', help='Subcommand to run\nmake: makes a database\
        \nquery: queries an existing database\nrandom: returns random books from a database.\
        \n\nFor more detailed help, run <command> -h.')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print ('Unrecognized command: %s'%(args.command))
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def make(self):
        parser = argparse.ArgumentParser(
            description='Record changes to the repository')
        # prefixing the argument with -- means it's optional
        parser.add_argument('fname', action='store')
        parser.add_argument('--range', '-r', action="store", nargs=2, dest="scan_range")
        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args = parser.parse_args(sys.argv[2:])

        if os.path.exists(args.fname):
            if input("File {} exists, overwrite(y/N)? ".format(args.fname)) == "y":
                conn, c = scp.create_db(args.fname)
            else:
                sys.exit(0)
        else:
            conn, c = scp.create_db(args.fname)
        if not args.scan_range:
            low = 0
            high = 50000
        else:
            low = args.scan_range[0]
            high = args.scan_range[1]
        for i in range(int(low), int(high)):
            if i%250 == 0 and i != 0:
                print("{} records processed".format(str(i)))
                conn.commit()
            d = scp.get_info(i)
            scp.add_to_table(d, c)
        conn.commit()

    def query(self):
        parser = argparse.ArgumentParser(
            description='Download objects and refs from another repository')
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument('fname', action='store')
        parser.add_argument('col', action='store')
        parser.add_argument('row', action='store')
        parser.add_argument('--output_dir', '-o', action='store')

        args = parser.parse_args(sys.argv[2:])

        conn, c = scp.connect_db(args.fname)
        #col = args.query[0]
        #row = args.query[1]
        lst = scp.query(c, args.col, args.row)
        if not args.output_dir:
            output_dir = "saves"
        else:
            output_dir = args.output_dir
        scp.retrieve_records(lst, output_dir)


    def random(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('fname', action='store')
        #parser.add_argument('mode', action='store')
        parser.add_argument('number', action='store')
        parser.add_argument('--output_dir', '-o', action='store')

        args = parser.parse_args(sys.argv[2:])

        if not args.output_dir:
            output_dir = "saves"
        else:
            output_dir = args.output_dir

        conn, c = scp.connect_db(args.fname)
        lst = c.execute("SELECT * FROM books ORDER BY RANDOM() LIMIT {}".format(int(args.number))).fetchall()
        scp.retrieve_records(lst, output_dir)



if __name__ == '__main__':
    ScrapeArgs()
