import os
fastas = os.listdir("dscript/fastas")
if ".DS_Store" in fastas:
    fastas.remove(".DS_Store")
# print(len(fastas))
    
deletes = []

for item in fastas:
    # print(fi[item].shape)
    with open(f'dscript/fastas/{item}','r') as in_file:
        l = in_file.read().splitlines()
        if len(l) != 4:
            deletes.append(item[:4])
        if len(l) == 4:
            if len(l[1]) >= 800 or len(l[3]) >= 800:
                # print(item)
                deletes.append(item[:4])
            if (len(l[1]) <= 50 or len(l[3]) <= 50):
                deletes.append(item[:4])
print(deletes)