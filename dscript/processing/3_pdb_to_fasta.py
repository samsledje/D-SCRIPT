"""
    # converting protein PDB files into fasta sequence format
"""
import Bio
from Bio import SeqIO
import os

entries = os.listdir('dscript/pdbsTEST')
if ".DS_Store" in entries:
    entries.remove(".DS_Store")
# print(entries)

# CONVERT ALL TO FASTAS
for i in range(0, len(entries)):
    # print(entries[i])
    if entries[i][0] != ".":
        with open(f'dscript/fastasTEST/{entries[i][:-4]}.fasta', 'w') as f:
            for record in SeqIO.parse(f"dscript/pdbsTEST/{entries[i]}", "pdb-seqres"):
                f.write(record.format("fasta-2line"))
        f.close()

# TEST A SINGLE
# with open(f'dscript/proteins/15c8.fasta', 'w') as f:
#         for record in SeqIO.parse(f"dscript/15c8.pdb", "pdb-seqres"):
#             # print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
#             # print(record.format("fasta"))
#             f.write(record.format("fasta-2line"))
# f.close()