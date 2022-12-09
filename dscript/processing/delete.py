import os

deletes = [
    "6MNR",
    "6PLK",
    "6CXF",
    "5Z2W",
    "6JEP",
    "4R8W",
    "6MNQ",
    "5IKC",
    "6OOR",
    "5FCU",
    "6B3M",
    "2K3U",
    "5U4K",
    "3IDY",
    "6THG",
    "6NZU",
    "6RHV",
    "3S5L",
    "6RHW",
    "5TTE",
    "6J5F",
    "5CW7",
    "3GD1",
    "6BCK",
    "5E9D",
    "6II4",
    "6II9",
    "6II8",
    "5WKO",
    "5VMM",
    "6UM5",
    "5Y11",
    "6J9L",
    "5KVD",
    "5GVI",
    "6J9M",
    "5KVG",
    "6PDX",
    "5KVF",
    "6JDJ",
    "6FPG",
    "3BGF",
    "5LCV",
    "6J11",
    "3FKU",
    "5TE6",
]
for item in deletes:
    if os.path.exists(f"dscript/pdb/{item}.pdb"):
        os.remove(f"dscript/pdb/{item}.pdb")

pdb_directory = "dscript"
files = os.listdir(f"{pdb_directory}")
if ".DS_Store" in files:
    files.remove(".DS_Store")
# print(len(files))
print(len(files))
