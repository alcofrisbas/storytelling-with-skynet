file_to_read = "RNN/data/train.txt"
file_to_write = "RNN/data/train1.txt"
with open(file_to_read, "r") as f1:
    lines = f1.readlines()
    mark = False
    with open(file_to_write, "w") as f2:
        for line in lines:
            if mark:
                line = "<GO>" + line
                line = line[:-1] + ". <OUT>\n"
            else:
                line = line[:-1] + ".\n"
            mark = not mark
            f2.write(line)
