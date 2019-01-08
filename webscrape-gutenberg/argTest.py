import argparse
import sys
import os
import scraper as scp


class ScrapeArgs(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', help='Subcommand to run')
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

    def query(self):
        parser = argparse.ArgumentParser(
            description='Download objects and refs from another repository')
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument('fname', action='store')
        parser.add_argument('col', action='store')
        parser.add_argument('row', action='store')
        parser.add_argument('-o', action)

    def random(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('fname', action='store')
        parser.add_argument('mode', action='store')
        parser.add_argument('number', action='store')

        args = parser.parse_args(sys.argv[2:])



if __name__ == '__main__':
    ScrapeArgs()
