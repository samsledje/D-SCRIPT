import os

files = os.listdir("dscript/pdbsTEST")
# fastas = os.listdir("dscript/fastasNEW")
# fastas.remove(".DS_Store")
if ".DS_Store" in files:
    files.remove(".DS_Store")
# print(len(files))
# print(files)

new_files = []
for item in files:
    new_files.append(item[:4].lower())
# print((new_files))

string = "'1r8o', '1nsg', '1pc8', '1ktp', '1puu', '1mlb', '1p6a', '1oo9', '1plg', '1jgv', '1jql', '1kfu', '1nld', '1q9k', '1kem', '1kel', '1jlt', '1qbl', '1mim', '1pqz', '1nlb', '1nl4', '1phn', '1jnn', '1qav', '1qbm', '1jnl', '1nme', '1op9', '1jtt', '1ppf', '1n7m', '1ncw', '1mh2', '1nbv', '1rf8', '1jv5', '1opg', '1n94', '1p69', '1p4q', '1kn4', '1oz7', '1on7', '1kcu', '1ngy', '1kn2', '1kcv', '1ngz', '1pum'"
new_string = ""
for item in string:
    # print(item)
    if item != "'" and item != " ":
        new_string += item[:4]

print(new_string)
