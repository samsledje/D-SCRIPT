import os
fastas = os.listdir("dscript/fastasNEW")
fastas.remove(".DS_Store")
# print(len(fastas))
    
deletes = []

for item in fastas:
    # print(fi[item].shape)
    with open(f'dscript/fastasNEW/{item}','r') as in_file:
        l = in_file.read().splitlines()
        if len(l) > 4:
            deletes.append(item[:4])
        if len(l[1]) >= 800 or len(l[3]) >= 800:
            deletes.append(item[:4].lower())
            
print(deletes)