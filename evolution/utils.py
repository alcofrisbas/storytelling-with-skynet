import struct

import struct

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def int_to_bin(num):
    return format(num, '032b')

def bin_to_int(binary):
    return int(binary, 2)




if __name__ == '__main__':
    print(float_to_bin(399.088758687))
    print(int_to_bin(399))
    print(bin_to_int("00000000000000000000000000000101"))
