import argparse
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    subparsers = parser.add_subparsers(help='types of A')

    m_parser = subparsers.add_parser("make")
    q_parser = subparsers.add_parser("query")

    args = parser.parse_args()
    print(args)
    # if args.action == "make":
    #     print("making")
    #     if args.range:
    #         print("file range: {}-{}".format(args.range[0], args.range[1]))
    #     else:
    #         print("no range specified, aborting")
    # elif args.action == "query":
    #     print("querying")
    #     if args.query:
    #         print(args.query)
    #     elif args.random_select:
    #         print(args.random_select)
    #     else:
    #         print("invalid query format")
    # elif args.action == "update":
    #     pass
    # else:
    #     print("'{}' is not a valid action".format(args.action))
    # print(args)
