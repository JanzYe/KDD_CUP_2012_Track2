
from constants import *
def check(input_path):
    fin = open(input_path)
    print(fin.readline())
    print(fin.readline())
    print(fin.readline())
    print(fin.readline())
    print(fin.readline())
    print(fin.readline())

    fin.close()

if __name__ == '__main__':
    check(DIR_PATH + "combined_mapped_test.txt")