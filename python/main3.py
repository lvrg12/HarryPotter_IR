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
    # for name in sorted(os.listdir("../dataset2/pdfs")):

    #     if( name == ".DS_Store" or name == "hp3_22.pdf" or name == "hp4_37.pdf"):
    #         continue

    #     N = N + 1
    #     f = open("../dataset2/pdfs/" + name, 'rb')
    #     pdfReader = PyPDF2.PdfFileReader(f)
    #     text = ""
    #     # print(name)
    #     for page in range(pdfReader.numPages):
    #         text = text + " " + pdfReader.getPage(page).extractText()
    #     doc[N] = pp.preprocess(text)

    # 2. TF and IDF
    corpus = {}
    # for d in doc:
    #     for w in doc[d]:
    #         if w in corpus:
    #             corpus[w][d] = corpus[w][d] + 1
    #         else:
    #             corpus[w] = {}
    #             for i in range(N):
    #                 corpus[w][i+1] = 0
    #             corpus[w][d] = 1


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
    # print(cl.clust(vector_space,5))

    # 4. P(c)
    with open('../locations.csv', newline='') as file:
        reader = csv.reader(file)
        classes = {}
        total = 0
        for row in reader:

            if( row[2] == "place" ):
                continue

            total = total + 1
            if row[2] in classes:
                classes[row[2]] = classes[row[2]] + 1
            else:
                classes[row[2]] = 1
    
    p_c = {}
    for c in classes:
        p_c[c] = classes[c] / total
        print( "P(" + c + ") =\t\t\t" + str(classes[c]) + "/" + str(total))

    print(p_c)

def normalized( vector ):

    sum = 0
    for c in vector:
        sum = sum + c

    if sum == 0:
        return vector
    else:
        return [ c/sum for c in vector ]


if __name__ == "__main__":
    main()