import sys
import scraper


def get_alpha(fname):
    with open(fname) as r:
        text = r.read()
    return sorted(list(set(list(text))))

if __name__ == '__main__':
    fname = sys.argv[1]
    print(get_alpha(fname))
