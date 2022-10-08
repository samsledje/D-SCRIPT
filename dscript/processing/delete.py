import os

deletes = []
for item in deletes:
    if os.path.exists(f"dscript/pdb/{item}.pdb"):
        os.remove(f"dscript/pdb/{item}.pdb")
