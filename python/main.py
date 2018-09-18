import os
import codecs
import preprocessing as pp

def main():

    for name in os.listdir("../dataset"):
        
        doc = codecs.open("../dataset/" + name, 'r', 'utf-8-sig').read()

        print(name)
        print("Tokenization\t" + str(len(pp.tokenize(doc))) )
        print("Lemmatizaton\t" + str(len(pp.lemmatize(doc))) )
        print("Stemming\t" + str(len(pp.stem(doc))) )

        break


if __name__ == "__main__":
    main()