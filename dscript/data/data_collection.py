"""
    # converting protein PDB files into fasta sequence format
"""
import Bio
from Bio import SeqIO
import os

entries = os.listdir('dscript/pdbs')

# CONVERT ALL
for i in range(0, len(entries)):
    if entries[i][0] != ".":
        with open(f'dscript/fastas/{entries[i][:-4]}.fasta', 'w') as f:
            for record in SeqIO.parse(f"dscript/pdbs/{entries[i]}", "pdb-seqres"):
                f.write(record.format("fasta-2line"))
        f.close()

# TEST A SINGLE
# with open(f'dscript/proteins/15c8.fasta', 'w') as f:
#         for record in SeqIO.parse(f"dscript/15c8.pdb", "pdb-seqres"):
#             # print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
#             # print(record.format("fasta"))
#             f.write(record.format("fasta-2line"))
# f.close()