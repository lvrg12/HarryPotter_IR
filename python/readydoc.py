import os

def reverse():

    for name in os.listdir("../dataset2/pdfs"):
        n_arr1 = name.split("_")
        n_arr2 = n_arr1[1].split(".")
        new_name = "../dataset2/pdfs/" + n_arr2[0]+"_"+n_arr1[0]+".pdf"

        os.rename("../dataset2/pdfs/" + name, new_name)

def chapterize():

    hp = "hp1"
    chapter = 0

    for name in sorted(os.listdir("../dataset2/pdfs")):
        c_hp = name.split("_")[0]

        if( c_hp == hp ):
            chapter = chapter + 1
        else:
            hp = c_hp
            chapter = 1

        new_name = "../dataset2/pdfs/" + hp + "_"+ str(chapter) + ".pdf"

        # print(new_name)

        os.rename("../dataset2/pdfs/" + name, new_name)

if __name__ == "__main__":
    reverse()
    # chapterize()