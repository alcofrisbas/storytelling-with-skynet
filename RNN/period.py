"""
A simple script for adding periods at the end of lines of the dataset
"""

with open("ptb.test.txt", "r") as f:
    with open("ptb.test1.txt", "w") as f1:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line += '.'
            line += '\n'
            f1.write(line)
