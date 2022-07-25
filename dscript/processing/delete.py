import os
 
# deletes = ["1B6H",  "2Z59", "2N73", "6W0V", "5Z2W", "6S3W", "7JOD", "7N1N", "7PKU", "1B6H", "5JYQ", "6GU0", "7JOE", "6E4D", "2Z59", "7B2B", "5U4K", "6KK3", "2N73", "7AA4", "6W9K", "7JQV", "6W9M", "7JOW"]
# new = []
# for item in deletes:
#     new.append(f"{item.lower()}.pdb")
# print(new)

deletes = ['5y79', '6hqb', '5weh', '5xat', '6giq', '4y7j', '5djq']
for item in deletes:
    if os.path.exists(f"dscript/pdbsNEW/{item}.pdb"):
        os.remove(f"dscript/pdbsNEW/{item}.pdb")
for item in deletes:
    if os.path.exists(f"dscript/fastasNEW/{item}.fasta"):
        os.remove(f"dscript/fastasNEW/{item}.fasta")

