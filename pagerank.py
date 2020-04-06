import os
import sys
import math
import numpy as np
from scipy.sparse import csc_matrix


def pageRank(G, d = .85, epslion = .0001):
    """
    G is a matrix represents outgoing edges between pages
    0,1 at i,j represents existence of j->i edge
    i.e. column j of G represents the outgoing edges of j

    d is damping factor
    1 - d is the probability to jump to random page

    epsilon the convergence parameter

    the formula is:
    rank = S * rank
    and with damping factor:
    rank = d * S * rank + (1 - d) / N
    """
    # num of nodes
    N = G.shape[0]

    # stochastic G
    sG = G / G.sum(axis=0)
    sG = (d * sG + (1 - d) / N)

    # init rank with uniform prob
    r = np.full(8, 1 / N)
    prev_r = 0

    # iterate until small diff between iterations
    while np.sum(np.abs(r - prev_r)) > epslion:
        prev_r = r.copy()
        r = np.dot(sG, r)

    # return normalized pagerank
    return r


def my_links():
    pages = ["bar ilan",
             "ramat gan",
             "library",
             "daniel hershkowitz",
             "judaism",
             "minister of education",
             "exact sciences",
             "engineering"]

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

    print("\nresults for 0.8")
    ranks = pageRank(links, d=.8)
    for (page,rank) in zip(pages,ranks):
        print(page + " " + str(rank))

    print("\nresults for 0.9")
    ranks = pageRank(links,d=.9)
    for (page,rank) in zip(pages,ranks):
        print(page + " " + str(rank))