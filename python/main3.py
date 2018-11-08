import os
import math
import csv
import unidecode
import PyPDF2
import preprocessing as pp
import term_frequency as tf
import tfidf
import vector_space_model as vsm
import clustering as cl

def main():

    # 1. Pre-processing
    doc = {}
    N = 0
    for name in sorted(os.listdir("../dataset2/pdfs")):

        if( name == ".DS_Store"):
            continue

        N = N + 1
        f = open("../dataset2/pdfs/" + name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(f)
        text = ""
        print(name)
        for page in range(pdfReader.numPages):
            text = text + " " + pdfReader.getPage(page).extractText()
        doc[N] = pp.preprocess(text)

    # 2. TF and IDF
    corpus = {}
    for d in doc:
        for w in doc[d]:
            if w in corpus:
                corpus[w][d] = corpus[w][d] + 1
            else:
                corpus[w] = {}
                for i in range(N):
                    corpus[w][i+1] = 0
                corpus[w][d] = 1


    # 2. Vector Space
    if not os.path.exists("tf_vector.csv"):
        with open('tf_vector.csv', 'w', newline='') as csv_td:
            writer = csv.writer(csv_td)
            writer.writerow(list(corpus.keys()))
            for i in range(N):
                d = int(i) + 1
                row = []
                for w in corpus:
                    row.append(corpus[w][d])
                writer.writerow(row)

    vector_space = []
    with open('tf_vector.csv', newline='') as csv_td:
        reader = csv.reader(csv_td)
        for row in reader:
            if row[0].isdigit():
                vector = [ int(c) for c in row ]
                vector_space.append(normalized(vector))

    # 3. Cosine Similarity
    if not os.path.exists("cos_sim.csv"):
        with open('cos_sim.csv', 'w', newline='') as csv_cs:
            writer = csv.writer(csv_cs)
            i = 0
            for v1 in vector_space:
                row = []
                for v2 in vector_space:
                    sim = round(vsm.cosine_similarity(v1,v2),3)
                    row.append(sim)
                writer.writerow(row)

    # 5. K-Means Clustering
    print(cl.clust(vector_space,5))

def normalized( vector ):

    sum = 0
    for c in vector:
        sum = sum + c

    if sum == 0:
        return vector
    else:
        return [ c/sum for c in vector ]



    

    # a) Term Frequency
    # print("Term Frequency")
    # for t in query:
    #     print(t, end="  ")
    #     for d in pos[t]:
    #         print( str(d) + ": " + str(tfidf.tf(t,d,pos)) + " ", end=" ")
    #     print()

    # arrtmp = [0] * 130
    # for t in query:
    #     print(t, end=",")
    #     for d in range(len(arrtmp)):
    #         if d in pos[t]:
    #             print( str(tfidf.tf(t,d,pos)), end="," )
    #         else:
    #             print( end="0," )
    #     print()

    # max_f = {}
    # for t in query:
    #     for d in pos[t]:
    #         max_f[t] = len(pos[t][d])



    # # a) Inverse Document Frequency
    # # print("\nInverse Document Frequency")
    # # for t in query:
    # #     print(t + "," + str( tfidf.idf(len(doc),t,pos) ) )

    # # 4. TF-IDF
    # # print("\nTF-IDF")
    # # for d in doc:
    # #     print( str( tfidf.tfidf(len(doc),d,query,pos) ) )

    # # 6. Vector Space
    # vector_space = {}
    # # print("\nVector Space")
    # for d in doc:
    #     vector_space[d] = vsm.vector(d,query,pos,len(doc),max_f)
    #     # print( str(d) + ": " + str( vector[d] ) )

    # # 7. Cosine Similarity
    # # print("\nCosine Similarity")
    # vector_space = vsm.normalize_vector(vector_space)

    # cos_sim_matrix = []
    # i = 0
    # for d in doc:
    #     cos_sim_matrix.append([])
    #     for d2 in doc:
    #         sim = round(vsm.cosine_similarity(vector_space[d],vector_space[d2]),3)
    #         cos_sim_matrix[i].append(sim)
    #         # print(sim, end=",")
    #     # print()
    #     i = i + 1

    # print(cl.clust(cos_sim_matrix,5))


if __name__ == "__main__":
    main()