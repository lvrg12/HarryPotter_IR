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

    # 2. Pre-processing
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

    # 3. TF and IDF
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


    # 3. Vector Space
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

    vocabulary = []
    tf_vector = []
    vector_space = []
    with open('tf_vector.csv', newline='') as csv_td:
        reader = csv.reader(csv_td)
        for row in reader:
            if row[0].isdigit():
                vector = [ int(c) for c in row ]
                tf_vector.append(vector)
                vector_space.append(normalized(vector))
            else:
                vocabulary = [ w for w in row ]

    # 4. P(c)
    doc_count = 1
    doc_classes = {}
    with open('../locations_training2.csv', newline='') as file:
        reader = csv.reader(file)
        classes = {}
        total = 0
        for row in reader:

            if( row[2] == "place" ):
                continue

            doc_classes[doc_count] = row[2]
            doc_count = doc_count + 1

            total = total + 1
            if row[2] in classes:
                classes[row[2]] = classes[row[2]] + 1
            else:
                classes[row[2]] = 1
    
    p_c = {}
    for c in classes:
        p_c[c] = classes[c] / total
        # print( "P(" + c + ") =\t\t\t" + str(classes[c]) + "/" + str(total))

    # 5. P(w|c)
    p_wc = {}
    class_word_count = {}

    # counting words in classes
    for c in classes:
        class_word_count[c] = 0
        for w in range(len(vocabulary)):
            for d in range(len(doc_classes)):
                if doc_classes[d+1] == c:
                    class_word_count[c] = class_word_count[c] + tf_vector[d][w]

    for c in classes:
        p_wc[c] = {}
        for w in range(len(vocabulary)):
            n = 0
            for d in range(len(doc_classes)):
                if doc_classes[d+1] == c:
                    n = n + tf_vector[d][w]
            p_wc[c][vocabulary[w]] = (n + 1) / ( class_word_count[c] + len(vocabulary) )
    
    p_wc_optimized = {}
    for c in p_wc:
        arr = []
        for w in p_wc[c]:
            if len(w) > 3:
                arr.append((w,p_wc[c][w]))
        arr = sorted(arr,key=lambda x: x[1],reverse=True)[:150]

        p_wc_optimized[c] = {}
        # print(c)
        for t in arr:
            # print(t[0])
            p_wc_optimized[c][t[0]] = t[1]
        # print()

        # print(p_wc_optimized)

    



    # 6. Posterior probabilities
    if not os.path.exists("pp.csv"):
        with open('pp.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            h1 = ['class','p(c)']
            h2 = [ w for w in vocabulary ]
            writer.writerow(h1 + h2)
            for c in classes:
                row1 = [c, p_c[c]]
                row2 = [ p_wc[c][w] for w in p_wc[c] ]
                writer.writerow(row1 + row2)

    # Classification
    test_doc = {}
    N = 0
    for name in sorted(os.listdir("../dataset2/dataset_test")):
        if( name == ".DS_Store" ):
            continue
        N = N + 1
        f = open("../dataset2/dataset_test/" + name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(f)
        text = ""
        for page in range(pdfReader.numPages):
            text = text + " " + pdfReader.getPage(page).extractText()
        test_doc[N] = pp.preprocess(text)

    test_doc_classes = {}
    N = 0
    with open('../locations_test2.csv', newline='') as file:
        reader = csv.reader(file)

        for row in reader:
            if( row[2] == "place" ):
                continue
            N = N + 1
            test_doc_classes[N] = row[2]

    # print("doc,house,hogwards,outside")
    header = "doc,"
    for c in classes:
        header = header + c + ","
    print(header)
    for d in test_doc:
        mlc = ""
        max_l = 0
        string = str(d)
        for c in classes:
            v = abs(math.log(p_c[c],1000) * multiplication( test_doc[d], p_wc[c] ))
            string = string + "," + str(v)
            if v > max_l:
                max_l = v
                mlc = c

        print(string)



def multiplication( d_corpus, c_corpus ):
    # d_corpus = list(set(d_corpus))
    product = 1
    for w in d_corpus:
        if w in c_corpus and c_corpus[w] > 0:
            product = product * math.log(c_corpus[w],1000)
            
    return product
    



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