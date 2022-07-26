import os

files = os.listdir("dscript/pdbsTEST")
# fastas = os.listdir("dscript/fastasNEW")
# fastas.remove(".DS_Store")
if ".DS_Store" in files:
    files.remove(".DS_Store")
print(len(files))

# new_files = []
# for item in files:
#     new_files.append(item[6:10])
# # print(new_files)

# string = "'6c3o', '5azd', '5kli', '2hg5', '3ag3', '3hzq', '5zzn', '4xtc', '6dho', '5cth', '6et5', '2f95', '6hum', '2wlo', '5xdq', '4ycr', '6eo1', '6hd1', '1l0n', '2yev', '5xtc', '5xth', '1vf5', '6nd1', '6fwz', '2m6i', '6c8h', '6hwh', '6dhp', '4p6v', '5mrw', '5b58', '1vf5', '5zov', '6bbi', '5xtd', '5xth', '4p9o', '6f2d', '4ri3', '6gyb', '6bhp', '4ryj', '1jb0', '4h44', '5zfv', '1ezv', '4gpo', '6drj', '5d3m', '4xto', '2vpz', '6f0k', '1kpl', '6a69', '1nek', '5kli', '6fkf', '6g2j', '5mg3', '5xtd', '5xtd', '6cvl', '6nd1', '5lki', '4p6v', '5vhz', '5bz3', '6hwh', '1l9j', '6hum', '5duo', '5o9h', '6giq', '6giq', '1vf5', '5xmj', '1vf5', '5mtf', '6csx', '1vf5', '2abm', '6mj2', '2onj', '6c6l', '4pv1', '5x41', '5j4n', '5ir6', '6hum', '5lzr', '4kjs', '5b0w', '6ezn', '4hw9', '6ctd', '5yve', '1nkz', '5but', '5eik', '2kdc', '5zlg', '5ir6', '1nen', '6et5', '6iu4', '5ofr', '6dhh', '5sv9', '6dhp', '6hzm', '4pi2', '6el1', '6mmn', '1nen', '6gct', '6c3p', '2m6x', '5u9w', '5zug', '6cxh', '6giq', '4pd4', '5wek', '5xdq', '5xth', '4v8k', '5jgp', '2a0l', '6cmc', '5tji', '4y7j', '4jrz', '3chx', '2hg5', '6dqj', '1ntk', '6iyc', '6cq9', '5u1y', '6fkf', '6hqb', '4xu6', '3wmm', '1nen', '4pi2', '6c3o', '1fft', '4qtn', '3jcf', '1vf5', '6hum', '6cfw', '5x87', '5yq7', '6g2j', '5uni', '6hwh', '6i53', '4q2g', '6hu9', '6f36', '5ogk', '6dhp', '5va3', '5tis', '3ogc', '1vf5', '4pd4', '6agf', '5ekp', '5wuf', '1ezv', '3j9v', '6c14', '1xqf', '1c8s', '5djq', '5vrf', '5y79', '5vot', '5jae', '5zji', '2d2c', '5ule', '3s3x', '2d2c', '6hu9', '5xnm', '3jqo', '4o9u', '5jnq', '1w5c', '5och', '5xat', '6hum', '4y6k', '6et5', '1l9j', '1fbb', '5vre', '6bvg', '3ir7', '2hg5', '6g94', '6dt0', '4ntb', '6dz7', '5d92', '1kqg', '5tj5', '6gy6', '2ibz', '6e1p', '6f36', '5n9y', '6f34', '5svk', 'vre.', '5yq7', '4p02', '5v78', '6ezn', '5cbh', '5kuf', '6hrb', '6iyc', '5xu1', '6dhp', '1cwq', '6cp7', '1nen', '2a79', '1ezv', '1l0n', '5aji', '6c70', '5x5y', '5weh', '6et5', '6iyc', '1kzu', '6qd5', '6bx5', '4bem', '6bpq', '5gup', '5v4s', '5iy5', '5uhs', '5xls', '5xnm', '1lda', '1w5c', '6b85', '5eul', '6hum', '5d6i', '5khs', '2wcd', '5djq', '5x9h', '6dhp', '4hw9', '6f0k', '6mhy', '3dh4', '6hum', '3chx', '4m8j', '5x87', '5vre'"
# new_string = ""
# for item in string:
#     # print(item)
#     if item != "'" and item !=" ":
#         new_string += item[:4]

# print(new_string)