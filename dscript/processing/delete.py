import os

string = "3srb1abt6Z8M1aot1b2a5D6J1a931b2b1a0n1kfu1ubo4m1j2omx1ubm1ubl1ai61ubh1ai71ai51ubk1ubj1ai41ajn1a2x1wul4U9H1ubr4U9I6XE31wuj1wuk1ajq2omw1wui1ubu1ubt1wuh1ajp1h2r3fju1abj4WKT1a7f1a9x"

deletes = []
for i in range(0, int(len(string) / 4)):
    i = i * 4
    deletes.append(string[i : i + 4])

print(deletes)

print(len(deletes))
for item in deletes:
    if os.path.exists(f"dscript/pdbs/{item}.pdb"):
        os.remove(f"dscript/pdbs/{item}.pdb")
for item in deletes:
    if os.path.exists(f"dscript/fastas/{item}.fasta"):
        os.remove(f"dscript/fastas/{item}.fasta")
