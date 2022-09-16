import os

deletes = []
print(len(deletes))
for item in deletes:
    if os.path.exists(f"dscript/pdbs_large/{item}.pdb"):
        os.remove(f"dscript/pdbs_large/{item}.pdb")
