import os
from preprocessing import prep

for file in os.listdir("./data"):
    data = prep.importData("./data/"+file, "\t")
    print(data)