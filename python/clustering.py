import statistics

def clust(matrix, k):
    m = len(matrix)

    n_cen = [10,27,45,73,110]
    # n_cen = [15,35,50,80,129]
    o_cen = [0] * k

    while o_cen != n_cen:
        print(n_cen)
        o_cen = [c for c in n_cen]

        # form K clusters
        cluster = {}
        for c in range(k):
            cluster[o_cen[c]] = []

        # assign vector to closest centroid
        for d in range(m):
            closest = o_cen[0]
            for cen in o_cen:
                if matrix[cen][d] > matrix[closest][d]:
                    closest = cen
            cluster[closest].append(d)
        
        # recompute centroid of each cluster
        for i in range(k):
            n_cen[i] = mean(cluster[o_cen[i]])

    return cluster


def mean( arr ):
    sum = 0
    for x in arr:
        sum = sum + x

    return int(sum/len(arr))
    # return int(statistics.median(arr))