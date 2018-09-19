import os
import math
import preprocessing as pp
import term_frequency as tf
import tfidf

def main():

    # 1. Pre-processing
    doc = {}
    for name in os.listdir("../dataset"):
        f = open("../dataset/" + name).read()
        doc[int(name.strip("hp").strip(".txt"))] = pp.preprocess(f)


    # 2. Query
    search = "magic owl wand"
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

            


    # print(pos)


if __name__ == "__main__":
    main()