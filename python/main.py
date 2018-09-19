import os
import math
import preprocessing as pp
import term_frequency as tf
import tfidf
import vector_space_model as vsm

def main():

    # 1. Pre-processing
    doc = {}
    for name in os.listdir("../dataset"):
        f = open("../dataset/" + name).read()
        doc[int(name.strip("hp").strip(".txt"))] = pp.preprocess(f)


    # 2. Query
    search = "magic owl wand study train school learn class fear teacher friend"
    query = search.split(" ")

    # 3. TF and IDF
    pos = {}
    for q in query:
        pos[q] = {}

        for d in tf.documents_with(q,doc):
            pos[q][d] = tf.document_positions(doc[d],q)

    # a) Term Frequency
    print("Term Frequency")
    for t in query:
        print(t, end="  ")
        for d in pos[t]:
            print( str(d) + ": " + str(tfidf.tf(t,d,pos)) + " ", end=" ")
        print()

    # a) Inverse Document Frequency
    print("\nInverse Document Frequency")
    for t in query:
        print(t + "  " + str( tfidf.idf(len(doc),t,pos) ) )

    # 4. TF-IDF
    print("\nTF-IDF")
    for d in doc:
        print( str(d) + ": " + str( tfidf.tfidf(len(doc),d,query,pos) ) )

    # 6. Vector Space
    vector = {}
    print("\nVector Space")
    for d in doc:
        vector[d] = vsm.vector(d,query,pos,len(doc))
        print( str(d) + ": " + str( vector[d] ) )

    # 7. Cosine Similarity
    print("\nCosine Similarity")
    vector = vsm.normalize_vector(vector)
    for d in doc:
        for d2 in doc:
            if d != d2:
                print(round(vsm.cosine_similarity(vector[d],vector[d2]),2), end=" ")
        print()                


if __name__ == "__main__":
    main()