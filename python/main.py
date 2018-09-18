import os
import preprocessing as pp

def main():

    for name in os.listdir("../dataset"):
        
        doc = open("../dataset/" + name).read()

        print(name)
        print("Tokenization\t" + str(len(pp.tokenize(doc))) )
        print("Normalization\t" + str(len(pp.normalize(doc))) )
        print("Lemmitizaton\t" + str(len(pp.lemmitize(doc))) )
        print("Stemming\t" + str(len(pp.stem(doc))) )

        break


if __name__ == "__main__":
    main()