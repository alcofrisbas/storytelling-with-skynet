from evo import *


a = {"a":1,"b":2}
b = {"a":5,"b":10}

def test_interpolation():
    print("=== interpolation ===")
    d = linear_interpolate(a,b,0.5)
    print(d)

def test_reproduce():
    print("=== reproduction ===")
    o1, o2 = reproduce(a,b)
    print(o1)
    print(o2)

if __name__ == '__main__':
    test_interpolation()
    test_reproduce()
