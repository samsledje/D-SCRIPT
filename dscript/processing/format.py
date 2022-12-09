import os

pdb_directory = "dscript/pdb"
files = os.listdir(f"{pdb_directory}")
if ".DS_Store" in files:
    files.remove(".DS_Store")
# print(len(files))
print(len(files))

with open("dscript/processing/pdb_ids.txt", "w+") as pdb_f:
    for item in files:
        pdb_f.write(f"{pdb_directory}/{item}")
        pdb_f.write("\n")
    # for item in files:
    #     pdb_f.write(f"{item[0:4]},")


# PDB BIND CODE
# new_pdb = []
# count = 0
# with open('dscript/processing/PDB_bind_files.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#         pdb = line.split(":")[1].split("\n")[0].split(" ")[1]
#         if f"{pdb.upper()}.pdb" not in files:
#             # print(f"{pdb.upper()}.pdb")
#             new_pdb.append(pdb)
#             count +=1
#     print(count)

# with open('dscript/processing/list_file.txt', 'w') as f:
#     for item in new_pdb:
#         f.write(f"{item.upper()},")
