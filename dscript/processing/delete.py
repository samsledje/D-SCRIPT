import os

# files = os.listdir("dscript/pdbsNEW")
# # fastas = os.listdir("dscript/fastasNEW")
# # fastas.remove(".DS_Store")
# files.remove(".DS_Store")
# print(len(files))
# print(len(set(files)))

deletes = ['4aw9', '4gts', '4kum', '4k90', '4cbx', '4ng9', '4gtt', '4bey', '4ob0', '4h0x', '4gtv', '4n80', '4asr', '4na9', '4n6n', '4cj1', '4asq', '4n6o', '4qo0', '4j4q', '4ehm', '4ish', '4qo2', '4jze', '4jzd', '4isi', '4gdx', '4c2a', '4qnz', '4ci6', '4k0a', '4g0n', '4afx', '4bb2', '4d7z', '4jyu', '4jyv', '4jzz', '4dig', '4h0t', '4am9', '4cbu', '4awa', '4nga', '4hpx', '4bdv']
for item in deletes:
    if os.path.exists(f"dscript/pdbsNEW/{item}.pdb"):
        os.remove(f"dscript/pdbsNEW/{item}.pdb")
for item in deletes:
    if os.path.exists(f"dscript/fastasNEW/{item}.fasta"):
        os.remove(f"dscript/fastasNEW/{item}.fasta")


# deletes = ['1tm4', '1y3b', '1tm5', '1rzx', '1tm3', '2arr', '1v5i', '1wpx', '1y3f', '1tmq', '2arq', '1smp', '1tkt', '1tgs', '2a93', '1u8c', '1w7x', '1y48', '1viw', '1u2u', '1twq', '1sdd', '1wht', '1to2', '1to1']
# short_del = []
# for item in deletes:
#     if item != [...] and item not in short_del:
#         short_del.append(item)

