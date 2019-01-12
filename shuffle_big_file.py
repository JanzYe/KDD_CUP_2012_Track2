import bisect
import tempfile
import random
import time
from constants import *

fin = open(DIR_PATH + 'training.txt')
output_path = DIR_PATH + 'shuffled_training.txt'

s = time.clock()

N = 149639105

print("Shuffling {0} lines...\n".format(N))
# random permutation
def random_permutation(N):
    l = list(range(N))
    for i, n in enumerate(l):
        r = random.randint(0, i)
        l[i] = l[r]
        l[r] = n
    return l

p = random_permutation(N)
ridx = [0] * N
files = []
mx = []

print("Computing list of temporary files\n")
for i, n in enumerate(p):
    pos = bisect.bisect_left(mx, n) - 1
    if pos == -1:
        files.insert(0, [n])
        mx.insert(0, n)
    else:
        files[pos].append(n)
        mx[pos] = n

P = len(files)
print("Caching to {0} temporary files\n".format(P))
fps = [tempfile.TemporaryFile(mode="w+") for i in range(P)]

for file_index, line_list in enumerate(files):
    print(file_index)
    for line in line_list:
        ridx[line] = file_index


# write to each temporal file
for i, line in enumerate(fin):
    if i % 100000 == 0:
        print(i)
        print(line)
    if i == 0:
        continue
    fps[ridx[i]].write(line)

for f in fps:
    f.seek(0)


print("Writing to the shuffled file\n")
output = open(output_path, 'w')
# write to the final shuffled file
output.write(','.join(headers) + '\n')
for i in range(N):
    line = fps[ridx[p[i]]].readline()
    output.write(line)

    if i % 100000 == 0:
        print(i)
        print(line)

e = time.clock()

print("Shuffling took an overall of {0} secs\n".format(e-s))