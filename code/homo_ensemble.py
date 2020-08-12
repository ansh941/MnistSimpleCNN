import numpy as np 
import argparse

cnt = 1
best = 10000
curr = 10000

p = argparse.ArgumentParser()
p.add_argument("--kernel_size", default=5, type=int)
args = p.parse_args()
KERNEL_SIZE = args.kernel_size

for i in range(10):
    for j in range(i+1,10):
        for k in range(j+1,10):
            w1 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, i)).astype(np.int)
            w2 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, j)).astype(np.int)
            w3 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, k)).astype(np.int)

            board = np.zeros((10000))
            board[w1] += 1
            board[w2] += 1
            board[w3] += 1
            board = board // 2
            curr = np.sum(board)
            if curr < best:
                best = curr
            print("%4d %4d %4d %4d %4d %4d"%(cnt, len(w1), len(w2), len(w3), curr, best))
            cnt += 1
