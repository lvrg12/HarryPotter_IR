import os
import preprocessing as pp

def main():

    doc = {}
    for name in os.listdir("../dataset"):
        
        f = open("../dataset/" + name).read()

        ignore = ["doc29.txt", "doc51.txt", "doc56.txt", "doc57.txt", "doc61.txt"]

        if( name in ignore ):
            continue

        # 1. Pre-processing
        doc[int(name.strip("doc").strip(".txt"))] = pp.preprocess(f)

    print(doc[2])

    # 2. Query
    query = "raven"

    # 3. Term Frequency
    




if __name__ == "__main__":
    main()