from Bio import SeqIO
from Bio import PDB
from Bio import pairwise2
import h5py
import numpy as np
import os
import pandas as pd

MAX_D = 25


def filter_chains(chain_list):
    chains_f = [[r for r in c if r.has_id("CA")] for c in chain_list]
    return chains_f


def residue_distance(res0, res1, max_d=25.0):
    diff_vector = res0["CA"].coord - res1["CA"].coord
    distance = np.sqrt(np.sum(diff_vector ** 2))

    return min(distance, max_d)


def calc_dist_matrix(
    chain0, chain1, seq0_long, seq0_short, seq1_long, seq1_short
):
    D = np.zeros((len(seq0_long), len(seq1_long)))
    D = D - 1

    ch0_it = iter(chain0)
    for (i, (res0L, res0S)) in enumerate(zip(seq0_long, seq0_short)):

        if res0S == "-":
            continue
        else:
            res0 = next(ch0_it)

        ch1_it = iter(chain1)
        for (j, (res1L, res1S)) in enumerate(zip(seq1_long, seq1_short)):

            if res1S == "-":
                continue
            else:
                res1 = next(ch1_it)

            D[i, j] = residue_distance(res0, res1, max_d=MAX_D)

    return D


def main():
    files = os.listdir("dscript/pdbsNEW")
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    for i in range(0, len(files)):
        files[i] = files[i][:4]

    hf_pair = h5py.File(f"data/paircmaps_train.h5", "w")
    count = 0
    total = 0
    for pdb_id in files:
        total += 1
        print(f"Total: {total}")
        print(pdb_id)

        pdb_file = f"dscript/pdbsNEW/{pdb_id}.pdb"

        seqres_recs = list(SeqIO.parse(pdb_file, "pdb-seqres"))
        # print(seqres_recs)
        atoms_recs = list(SeqIO.parse(pdb_file, "pdb-atom"))
        # print(atoms_recs)

        seqs_long = seqres_recs[:2]
        # print(seqs_long)
        seqs_short = atoms_recs[:2]
        # print(seqs_short)

        structure = PDB.PDBParser().get_structure(pdb_id, pdb_file)
        chains = list(structure.get_chains())[:2]
        # print(chains)
        chains_filtered = filter_chains(chains)
        # print(chains_filtered)

        seq0_long = seqs_long[0].seq
        # print(len(seq0_long))
        seq0_short = seqs_short[0].seq
        # print(len(seq0_short))
        chain0 = chains_filtered[0]
        # print((chain0))
        # print(len(chain0))

        seq1_long = seqs_long[1].seq
        # print(len(seq1_long))
        seq1_short = seqs_short[1].seq
        # print(len(seq1_short))
        chain1 = chains_filtered[1]
        # print((chain1))
        # print(len(chain1))

        if len(seq1_short) > len(seq1_long) or len(seq0_short) > len(
            seq0_long
        ):
            count += 1

            align0 = pairwise2.align.globalxx(seq0_long, seq0_short)
            print(pairwise2.format_alignment(*align0[0]))

            align1 = pairwise2.align.globalxx(seq1_long, seq1_short)
            print(pairwise2.format_alignment(*align1[0]))

        if len(chain0) == len(seq1_long) or len(chain0) == len(seq1_short):
            temp = chain1.copy()
            chain1 = chain0
            chain0 = temp
            print(len(chain0))
            print(len(chain1))

        align0 = pairwise2.align.globalxx(seq0_long, seq0_short)
        # print(pairwise2.format_alignment(*align0[0]))

        align1 = pairwise2.align.globalxx(seq1_long, seq1_short)
        # print(pairwise2.format_alignment(*align1[0]))

        D = np.zeros((len(seq0_long), len(seq1_long)))
        D = D - 1
        # print(D.shape)

        seq0_long_f = align0[0].seqA
        # print((seq0_long))
        # print((seq0_long_f))
        seq0_short_f = align0[0].seqB
        # print((seq0_short))
        # print((seq0_short_f))
        seq1_long_f = align1[0].seqA
        # print(seq1_long)
        # print(seq1_long_f)
        seq1_short_f = align1[0].seqB
        # print((seq1_short))
        # print((seq1_short_f))

        # print(seq0_long_f)
        # print(seq0_short_f)
        # print(seq1_long_f)
        # print(seq1_short_f)

        # print(seqs_long[0].seq)
        # print(chain0)

        # print(len(chain0))
        # print(len(chain1))

        # ch0_it = iter(chain0)
        # for (i, (res0L, res0S)) in enumerate(zip(seq0_long_f, seq0_short_f)):
        #     print((i, (res0L, res0S)))
        #     if res0S == "-" or res0L == "-":
        #         continue
        #     else:
        #         res0 = next(ch0_it)

        #     ch1_it = iter(chain1)
        #     for (j, (res1L, res1S)) in enumerate(zip(seq1_long_f, seq1_short_f)):

        #         if res1S == "-" or res1L == "-":
        #             continue
        #         else:
        #             res1 = next(ch1_it)

        #         D[i,j] = residue_distance(res0, res1, max_d = MAX_D)

        # D = calc_dist_matrix(chain0, chain1, seq0_long, seq0_short, seq1_long, seq1_short)
        # print(D.shape)
        # df = pd.DataFrame(D, index=list(seq0_long), columns=list(seq1_long))
        # print(df)

        # print(f'{pdb_id}:{str(chains[0].get_id())}x{pdb_id}:{str(chains[1].get_id())}')
        # hf_pair.create_dataset(f'{pdb_id}:{str(chains[0].get_id())}x{pdb_id}:{str(chains[1].get_id())}', data=D)
        print(f"Atom > Seqres: + {count}")
    print(count)


if __name__ == "__main__":
    main()
