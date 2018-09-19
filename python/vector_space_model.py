import tfidf
import math
import numpy

def vector_length(v):

    length = 0
    for c in v:
        length = length + c**2
    length = math.sqrt(length)

    return length

def vector(d,query,pos,N):

    vector = [0] * len(query)
    for i in range(len(query)):
        vector[i] = tfidf.tf_weight(d,query[i],pos) * tfidf.idf_weight(N,query[i],pos)

    return vector

def normalize_vector(vector):

    for v in vector:
        for i in range(len(vector[v])):
            if vector_length(vector[v]) != 0:
                vector[v][i] = vector[v][i] / vector_length(vector[v])

    return vector

def cosine_similarity(v1,v2):
    cs = numpy.dot(v1,v2) / ( vector_length(v1) * vector_length(v2) )
    return cs