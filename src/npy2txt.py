import numpy as np

J_all = np.load('data/sk30Jij.npy')

J = J_all[0]

np.savetxt('exhaustive_search/data/sk30Jij.txt', J, fmt='%.6f')
