import tfidf
import math
import numpy

def vector_length(v):

    length = 0
    for c in v:
        length = length + c**2
    length = math.sqrt(length)

    return length

def vector(d,query,pos,N,max_f):

    vector = [0] * len(query)
    i = 0
    for t in query:
        if d in pos[t]:
            vector[i] = len(pos[t][d]) # / max_f[t]  * tfidf.idf_weight(N,t,pos)
        else:
            vector[i] = 0
        i = i + 1

    return vector

def normalize_vector(vector):
    for v in vector:
        for i in range(len(vector[v])):
            if vector_length(vector[v]) != 0:
                vector[v][i] = vector[v][i] / vector_length(vector[v])

    return vector

def cosine_similarity(v1,v2):
    div = vector_length(v1) * vector_length(v2)
    if div == 0:
        cs = 0
    else:
        cs = numpy.dot(v1,v2) / div
    return cs