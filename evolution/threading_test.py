import threading
from queue import Queue
import time

def testThread(num):
    print(num)

if __name__ == "__main__":
    for i in range(5):
        t = threading.Thread(target=testThread, args=(i,))
        t.start()
