import os
 
# deletes = ["1B6H",  "2Z59", "2N73", "6W0V", "5Z2W", "6S3W", "7JOD", "7N1N", "7PKU", "1B6H", "5JYQ", "6GU0", "7JOE", "6E4D", "2Z59", "7B2B", "5U4K", "6KK3", "2N73", "7AA4", "6W9K", "7JQV", "6W9M", "7JOW"]
# new = []
# for item in deletes:
#     new.append(f"{item.lower()}.pdb")
# print(new)

# deletes = ['1cf4', '1irs', '1e0a', '1j4p', '1j4k', '1sdx', '2m6i', '1k2n', '3ugw', '1shc', '1z3m', '1b3g', '5H7P', '1l0a', '2a3i', '4ZXL', '1dow', '5D7E', '1oo4', '2n8j', '5J7J', '4P6W', '4fjp', '2r02', '1z3p', '5GXW', '1f3r', '1htr', '1b2h', '4R8T', '2rod', '2z7f', '1ym0', '1k2m', '2rs9', '1z3l', '1b3f', '1zbc', '1hag', '4XOE', '4SRN', '2yka', '2roc', '1ees', '1d5s', '1d5h', '2roz', '1n7t', '2r05', '1cqg', '1j4q', '2n2h', '4PID', '2rnw', '1f1w', '2kdc', '1fu5', '2aos', '1wlp', '1qg1', '1tce', '1ppe', '1fev', '3n3x', '1smh', '4WB8', '1t01', '2rmx', '2r03', '1d7q', '1j4l', '1b8q', '1zbk', '2vu8', '1jsp', '2pon', '2rny', '2my3', '1ddm', '3llz', '2yu7', '2n0y', '5APR', '2oq1', '2ror', '2qos', '1k3n', '3n2d', '1uoo', '1o0p', '2a0t', '4Q5U', '2uzv', '1l0n', '1uop', '1zbv', '1b3h', '2xjy', '1k3q', '1ssc', '3u8q', '1d5d', '2rqw', '2xs4', '1r8u', '1d4t', '5duo', '2uzt', '1xr0', '4TT2', '1opi', '1qmb', '1b32', '2my2', '1jm4', '1fhr', '2phg', '2rnx', '2m6x', '3h8k', '1zl1', '1wkw', '1fpr', '3m7u', '1b9j', '2uzu', '1cqh', '2n9p', '1t37', '1zbw', '1b4z', '2uzw', '1d5e', '1tg4', '4W8P']
deletes = ['7KLR', '4bs3', '4lnp', '6MHA', '1kbh', '1rgj', '6KBV', '6ES5', '2ofq', '5XBO', '2n2v', '1om2', '1i8h', '5Y20', '6ES7', '1hoy', '2ain', '6KBO', '2mv7', '7KN0', '6MGN', '5X9X', '1q69', '6B7G', '1b18', '2rqg', '7OJ9', '5L85', '2n4q', '1l2z', '1jh4', '4Q6H', '2rpn', '5D5E', '1s5q', '5MIZ', '7E0B', '1ncp', '7JQ8', '1kuz', '1hv2', '4NIB', '6X4X', '5J6Z', '1u0i', '1eci', '2rn5', '1bph', '6XMN', '6ES6', '1iog', '1q68', '1s7p', '1oqp', '6F55', '3t2a', '1kmf', '1pnb', '1k4u', '2n2w', '1sdb', '1hls', '2zpp', '1bxp', '1xgl', '1s5r', '6K59', '1q5w', '6QXZ', '1bzv', '1cph', '1b19', '1u38', '2rsy', '2n9y', '2rm0', '1zsg', '1jmq', '6BGH', '4i5y', '6VES', '4iuz', '1sse', '1t1k', '1uel', '1hiq', '1t1p', '6KH8', '1b2g', '1qh2', '1i5h', '5LIS', '3u4n', '1sf1', '1b2e', '4m5s', '1his', '1b17', '2rqh', '5YC4', '1q0w', '2a4j', '2mvc', '1mhi', '5VIZ', '1hui', '1vkt', '1otr', '1i8g', '4YYP', '1sb0', '1bon', '6KHA', '1bh9', '2mwn', '1pmx', '1tkq', '1io6', '1zgx', '6Q00', '2rly', '6Q8Q', '7QDW', '2n55', '5D53', '2pq4', '5EN9', '1b2d', '1g1e', '9INS', '5GWM', '1sjt', '4ihn', '5ENA', '6C0A', '1jbd', '5D54', '4i5z', '7A2Y', '7CWH', '3i40', '1kdx', '2os6', '7JYN', '1pd7', '1mv0', '5MHD', '1jco', '6KH9', '1b2f', '1t1q', '1k3m', '6K5R', '1bom', '1d5g', '7NHU', '1hit', '1jgn', '5YC3', '2mur', '5D52', '6S3F', '1kup', '1mhj', '2mzd', '2pku', '6NSX', '7F7X', '2mvd', '2wfu', '1dph', '2n2x', '4iyd', '2pdz', '1qfn', '7AC4', '4iyf', '5I22', '3i3z', '5MWQ', '6K5T', '2rol', '7EDP', '1ioh', '1bh8']
print(len(deletes))
for item in deletes:
    if os.path.exists(f"dscript/pdbs/{item}.pdb"):
        os.remove(f"dscript/pdbs/{item}.pdb")
for item in deletes:
    if os.path.exists(f"dscript/fastas/{item}.fasta"):
        os.remove(f"dscript/fastas/{item}.fasta")

