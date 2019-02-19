import multiprocessing

def spawn(num):
    print(num)

if __name__ == "__main__":
    for i in range(5):
        p = multiprocessing.Process(target=spawn,args=(i,))
        p.start()
        p.join()
