import os
import preprocessing as pp
import term_frequency as tf

def main():

    # 1. Pre-processing
    doc = {}
    for name in os.listdir("../dataset"):
        f = open("../dataset/" + name).read()
        doc[int(name.strip("hp").strip(".txt"))] = pp.preprocess(f)


    # 2. Query
    search = "magic owl wand"
    query = search.split(" ")


    # 3. Term Frequency
    pos = []
    for q in query:
        pos.append({})
        x = len(pos) - 1
        pos[x][q] = 0

        for d in tf.documents_with(q,doc):
            pos[x][q] = pos[x][q] + 1
            pos[x][d] = tf.document_positions(doc[d],q)

    print(pos)


if __name__ == "__main__":
    main()