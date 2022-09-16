import os

pdb_directory = "dscript/pdbs"
files = os.listdir(f"{pdb_directory}")
if ".DS_Store" in files:
    files.remove(".DS_Store")
# print(len(files))

with open("dscript/processing/pdb_ids.txt", "w+") as pdb_f:
    for item in files:
        pdb_f.write(f"{pdb_directory}/{item}")
        pdb_f.write("\n")
