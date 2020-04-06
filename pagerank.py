import os
import sys
import math
import numpy as np
from scipy.sparse import csc_matrix

def pageRank(G, s = .85, maxerr = .0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums==0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0,n):
            # inlinks of state i
            Ai = np.array(A[:,i].todense())[:,0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )

    # return normalized pagerank
    return r/float(sum(r))


def my_links():
    pages = ["https://en.wikipedia.org/wiki/Bar-Ilan_University",
             "https://en.wikipedia.org/wiki/Ramat_Gan",
             "https://en.wikipedia.org/wiki/Library",
             "https://en.wikipedia.org/wiki/Daniel_Hershkowitz",
             "https://en.wikipedia.org/wiki/Judaism",
             "https://en.wikipedia.org/wiki/Ministry_of_Education_(Israel)",
             "https://en.wikipedia.org/wiki/Exact_sciences",
             "https://en.wikipedia.org/wiki/Engineering"]

    links = np.array([[0,1,1,1,0,0,1,0],
                      [1,0,1,1,0,0,0,0],
                      [1,1,0,0,0,0,0,0],
                      [1,0,0,0,0,1,0,0],
                      [1,0,1,1,0,0,0,0],
                      [1,0,0,1,0,0,0,0],
                      [1,0,1,1,0,1,0,1],
                      [1,0,0,1,1,0,1,0]])

    return links, pages


if __name__ == '__main__':
    # calc pagerank with two damping values
    links,pages = my_links()
    ranks = pageRank(links, s=.8)
    for (page,rank) in zip(pages,ranks):
        print(page + " " + str(rank))

    ranks = pageRank(links,s=.9)
    for (page,rank) in zip(pages,ranks):
        print(page + " " + str(rank))