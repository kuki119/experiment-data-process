import AFS
import numpy as np

if __name__ == '__main__':

    bound = np.tile([[-600], [600]], 25)
    afs = AFS(60, 25, bound, 500, [0.001, 0.0001, 0.618, 40])
    afs.solve()