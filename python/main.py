import os
import preprocessing as pp

def main():

    for name in os.listdir("../dataset"):
        
        doc = open("../dataset/" + name).read()

        ignore = ["doc29.txt", "doc51.txt", "doc56.txt", "doc57.txt", "doc61.txt"]

        if( name in ignore ):
            continue

        print( name + "\t" + str(len(pp.preprocess(doc))) )
        
        #break


if __name__ == "__main__":
    main()