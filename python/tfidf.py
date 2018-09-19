import math

def tf(t,d,pos):
    return len(pos[t][d])

def idf(N,t,pos):
    if( len(pos[t]) == 0 ):
        return 0
    else:
        return math.log10(N/len(pos[t]))

def tf_weight(d,t,pos):
    if( d in pos[t] ):
        w = 1 + math.log10( tf(t,d,pos) )
    else:
        w = 0

    return w

def idf_weight(N,t,pos):
    w = idf(N,t,pos)

    return w

def tfidf(N,d,query,pos):
    w = 0

    for t in query:
        w = w + tf_weight(d,t,pos) * idf_weight(N,t,pos)

    return w