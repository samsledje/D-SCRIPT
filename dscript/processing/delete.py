import os

deletes = []
for item in deletes:
    if os.path.exists(f"dscript/pdb/{item}.pdb"):
        os.remove(f"dscript/pdb/{item}.pdb")

pdb_directory = "dscript"
files = os.listdir(f"{pdb_directory}")
if ".DS_Store" in files:
    files.remove(".DS_Store")
# print(len(files))
print(len(files))
