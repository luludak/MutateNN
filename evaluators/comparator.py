from math import *
from decimal import Decimal
 
# Using code from https://github.com/saimadhu-polamuri/DataAspirant_codes/tree/master/Similarity_measures
# With respect to owner copyrights and license.

def euclidean_distance(x,y):

    """ return euclidean distance between two lists """

    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y):

    """ return manhattan distance between two lists """

    return sum(abs(a-b) for a,b in zip(x,y))

def minkowski_distance(x,y,p_value):

    """ return minkowski distance between two lists """

    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
        p_value)

def nth_root(value, n_root):

    """ returns the n_root of an value """

    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def cosine_similarity(x,y):

    """ return cosine similarity between two lists """

    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def square_rooted(x):

    """ return 3 rounded square rooted value """

    return round(sqrt(sum([a*a for a in x])),3)

def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

# Implementation taken from https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
# With respect to owner copyrights and license.

# TODO: This implementation is untested - test extensively or use a more reliable computation method.
def rbo(list1, list2, p=0.9):
    list1 = list(list1)
    list2 = list(list2)
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2))/i
        term = pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)
    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)
    return ((float(x_k)/k) * pow(p, k)) + ((1-p)/p * summation)
