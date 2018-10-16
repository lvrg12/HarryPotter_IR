import statistics
import vector_space_model as vsm

def clust(vectors, k):

    n_cen = [vectors[25],vectors[50],vectors[75],vectors[100],vectors[125]]
    # n_cen = [15,35,50,80,129]
    o_cen = [0] * k

    while o_cen != n_cen:
        o_cen = [c for c in n_cen]

        # form K clusters
        cluster = {}
        for c in range(k):
            cluster[c] = []

        # assign vector to closest centroid
        for v in vectors:
            closest = o_cen[0]
            for c in o_cen:
                if vsm.cosine_similarity(v,c) > vsm.cosine_similarity(v,closest):
                    closest = c
            cluster[o_cen.index(closest)].append(vectors.index(v))
        
        # recompute centroid of each cluster
        for i in range(k):
            n_cen[i] = average(cluster[i],vectors)

    return cluster


def average( indexes, vectors ):
    n = len(indexes)
    centroid = [0] * len(vectors[0])

    for c in range(len(vectors[0])):
        sum = 0
        for index in indexes:
            sum = sum + vectors[index][c]
        if n == 0:
            centroid[c] = 0
        else:
            centroid[c] = sum/n

    return centroid