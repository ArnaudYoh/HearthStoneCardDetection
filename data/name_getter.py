from os import listdir
from os.path import isfile, join

def main():
    mypath = "./images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    with open('trainlist.txt', 'w') as traintext, open('testlist.txt', 'w') as testtext:
        for f in onlyfiles:
            name, ext = f.split(".")
            if name.isdigit():
                testtext.write(name +"\n")
            else:
                traintext.write(name + "\n")


if __name__ == "__main__":
    main()