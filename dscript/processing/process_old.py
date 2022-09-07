from Bio import SeqIO
from Bio import PDB
from Bio import pairwise2
import h5py
import numpy as np
import os
import pandas as pd
import csv
import argparse

MAX_D = 25


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """
    data_grp = parser.add_argument_group("Data Processing files")

    data_grp.add_argument(
        "--pdb_directory",
        required=True,
        help="name of pdb directory containing downloaded pdb files",
    )
    data_grp.add_argument(
        "--test", required=True, help="list of validation/testing pairs"
    )
    data_grp.add_argument(
        "--embedding",
        required=True,
        help="h5py path containing embedded sequences",
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_true",
        help="data is automatically augmented by adding (B A) for all pairs (A B). Set this flag to not augment data",
    )

    return parser


def inputs():
    """
    Requires user to input arguments for:
        name of pdb directory
        desired contact map dataset name
        desired fasta output name
        desired tsv file name
    """
    pdb_directory = input("Name of directory containing pdb files: ")
    files = os.listdir(f"dscript/{pdb_directory}")
    h5_name = input("Name of output H5 (for contact maps): ")
    hf_pair = h5py.File(f"data/{h5_name}.h5", "w")
    fasta_name = input("Name of output fasta (for sequences): ")
    tsv_name = input("Name of output tsv (for PPIs): ")
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    for i in range(0, len(files)):
        files[i] = files[i][:4]
    return [pdb_directory, files, fasta_name, tsv_name, hf_pair]


def get_sequences(pdb_directory, pdb_id):
    """
    Gets atom and seqres sequences from the pdb file.

    :param pdb_directory: name of directory containing pdb
    :type version: string
    :param pdb_id: name of pdb file
    :type version: string
    :return: list containing pdb_file, sequence from seq-res, and sequence from atom
    :rtype: list
    """
    pdb_file = f"dscript/{pdb_directory}/{pdb_id}.pdb"
    seqres_recs = list(SeqIO.parse(pdb_file, "pdb-seqres"))
    atoms_recs = list(SeqIO.parse(pdb_file, "pdb-atom"))
    seqs_long = seqres_recs[:2]
    seqs_short = atoms_recs[:2]
    return [pdb_file, seqs_long, seqs_short]


def check_chains(pdb_id, pdb_file, pdb_delete):
    """
    Gets atom and seqres sequences from the pdb file.

    :param pdb_id: name of pdb file
    :type version: string
    :param pdb_file: path to pdb file
    :type version: string
    :param pdb_delete: list of pdbs that don't satisfy length conditions
    :type version: list
    :return: list containing a list of chains and a list of pdbs that don't satisfy length conditions
    :rtype: list
    """
    structure = PDB.PDBParser().get_structure(pdb_id, pdb_file)
    chains = list(structure.get_chains())
    # if len(chains) > 2:
    #     pdb_delete.append(pdb_id)
    #     return None
    chains = list(structure.get_chains())[:2]
    if (
        len(chains[0]) > 800
        or len(chains[1]) > 800
        or len(chains[0]) < 50
        or len(chains[1]) < 50
    ):
        pdb_delete.append(pdb_id)
        return None
    return [chains, pdb_delete]


def make_tsv(pdb_id, chains, name):
    """
    Creates/appends protein chains to a tsv file.

    :param pdb_id: pdb formatted name of protein
    :type version: string
    :param chains: list of chains in protein
    :type version: list
    :param name: name of tsv file
    :type version: string
    """
    with open(f"data/{name}.tsv", "a") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        prot1 = f"{pdb_id.upper()}:{str(chains[0].get_id()).upper()}"
        prot2 = f"{pdb_id.upper()}:{str(chains[1].get_id()).upper()}"
        print((prot1, prot2))
        tsv_writer.writerow([prot1, prot2, "1"])


def make_fasta(pdb_id, seqs_long, fasta_name):
    """
    Creates/appends protein sequences to a fasta file.

    :param pdb_id: pdb formatted name of protein
    :type version: string
    :param seqs_long: sequences for 2 chains derived from seq-res
    :type version: list
    :param fasta_name: name of fasta file
    :type version: string
    """
    with open(f"data/{fasta_name}.fasta", "a") as f:
        for record in seqs_long:
            f.write(record.format("fasta-2line"))
            # None


def filter_chains(chain_list):
    """
    Filters chains to remove all HETATM (non-amino acid) residues.

    :param chain_list: pass in list of original chains
    :type version: list
    :return: list of filtered chains
    :rtype: list
    """
    chains_f = [[r for r in c if r.has_id("CA")] for c in chain_list]
    return chains_f


def split_sequences(seqs_long, seqs_short, chains_filtered):
    """
    Gets actual sequences (long = seq-res) and (short = atom) for the two chains

    :param seqs_long: list of seq-res sequences for chains 0 and 1
    :type version: string
    :param seqs_short: list of atom sequences for chains 0 and 1
    :type version: string
    :param seqs_short: list of filtered chains
    :type version: list
    :return: list of seq-res sequence for chain0, atom sequence for chain0, seq-res sequence for chain1, atom sequence for chain1
    :rtype: list
    """
    seq0_long = seqs_long[0].seq
    seq0_short = seqs_short[0].seq
    chain0 = chains_filtered[0]
    seq1_long = seqs_long[1].seq
    seq1_short = seqs_short[1].seq
    chain1 = chains_filtered[1]
    return [seq0_long, seq0_short, chain0, seq1_long, seq1_short, chain1]


def chain_switch(seq0_long, seq0_short, seq1_long, seq1_short, chain0, chain1):
    """
    Checks if chains and sequences were reversed while parsing
    (i.e. chain0 corresponds to sequence 1, and chain1 corresponds to sequence 0).
    If so, flip them.

    :param seq0_long: seq-res sequence for chain 0
    :type version: string
    :param seq0_short: atom sequence for chain 0
    :type version: string
    :param seq1_long: seq-res sequence for chain 1
    :type version: string
    :param seq1_short: atom sequence for chain 1
    :type version: string
    :param chain0: chain 0
    :type version: Chain object
    :param chain1: chain 1
    :type version: Chain object
    :return: list containing chain0, chain1 (possibly now reversed)
    :rtype: list
    """
    if (len(chain0) != len(seq0_long) or len(chain0) != len(seq0_short)) and (
        len(chain0) == len(seq1_long) or len(chain0) == len(seq1_short)
    ):
        temp = chain1.copy()
        chain1 = chain0
        chain0 = temp
    return [chain0, chain1]


def get_aligned_seqs(seq0_long, seq0_short, seq1_long, seq1_short):
    """
    Gets sequence0 and sequence1 after sequence alignment (containing "-").

    :param seq0_long: seq-res sequence for chain 0
    :type version: string
    :param seq0_short: atom sequence for chain 0
    :type version: string
    :param seq1_long: seq-res sequence for chain 1
    :type version: string
    :param seq1_short: atom sequence for chain 1
    :type version: string
    :return: list of aligned seq-res sequence0, aligned atom sequence0, aligned seq-res sequence1, aligned atom sequence1
    :rtype: list
    """
    align0 = pairwise2.align.globalxx(seq0_long, seq0_short)
    align1 = pairwise2.align.globalxx(seq1_long, seq1_short)
    # print(pairwise2.format_alignment(*align0[0]))
    # print(pairwise2.format_alignment(*align1[0]))
    seq0_long_f = align0[0].seqA
    seq0_short_f = align0[0].seqB
    seq1_long_f = align1[0].seqA
    seq1_short_f = align1[0].seqB
    return [seq0_long_f, seq0_short_f, seq1_long_f, seq1_short_f]


def residue_distance(res0, res1, max_d=25.0):
    """
    Calculates Euclidean distance between two amino acid residues, with a maximum distance threshold.

    :param res0: first residue
    :type version: Residue Object
    :param res1: second residue
    :type version: Residue Object
    :param max_d: distance
    :type version: float
    :return: distance between the two residues
    :rtype: Numpy float
    """
    diff_vector = res0["CA"].coord - res1["CA"].coord
    distance = np.sqrt(np.sum(diff_vector ** 2))
    return min(distance, max_d)


def calc_dist_matrix(
    pdb_id,
    chain0,
    chain1,
    seq0_long,
    seq1_long,
    seq0_long_f,
    seq0_short_f,
    seq1_long_f,
    seq1_short_f,
    errors,
):
    """
    Generates contact map between two chains using sequence alignment.

    :param pdb_id: pdb formatted name of protein
    :type version: string
    :param chain0: first chain
    :type version: Chain Object
    :param chain1: second chain
    :type version: Chain Object
    :param seq0_long: original sequence for chain 0
    :type version: Bio.Seq.Seq
    :param seq1_long: original sequence for chain 1
    :type version: Bio.Seq.Seq
    :param seq0_long_f: long alignment sequence for chain 0
    :type version: string
    :param seq0_short_f: short alignment sequence for chain 0
    :type version: string
    :param seq1_long_f: long alignment sequence for chain 1
    :type version: string
    :param seq1_short_f: short alignment sequence for chain 1
    :type version: string
    :return: generated distance matrix between two chains
    :rtype: Numpy matrix
    """
    D = np.zeros((len(seq0_long), len(seq1_long)))
    D = D - 1
    ch0_it = iter(chain0)
    x = -1
    y = 0
    for (i, (res0L, res0S)) in enumerate(zip(seq0_long_f, seq0_short_f)):
        if res0S == "-":
            x += 1
            continue
        if res0L == "-":
            continue
        else:
            x += 1
            try:
                res0 = next(ch0_it)
            except StopIteration:
                if pdb_id not in errors:
                    errors.append(pdb_id)
            # res0 = next(ch0_it)

        ch1_it = iter(chain1)
        for (j, (res1L, res1S)) in enumerate(zip(seq1_long_f, seq1_short_f)):
            if res1S == "-":
                y += 1
                continue
            if res1L == "-":
                continue
            else:
                try:
                    res1 = next(ch1_it)
                except StopIteration:
                    if pdb_id not in errors:
                        errors.append(pdb_id)
                # res1 = next(ch1_it)
                D[x, y] = residue_distance(res0, res1, max_d=MAX_D)
                y += 1
        y = 0
    # df = pd.DataFrame(D, index=list(seq0_long), columns=list(seq1_long))
    # print(df)
    return [D, errors]


def delete(pdb_delete, pdb_directory):
    """
    Deletes all files in pdb_delete from pdb_directory.

    :param pdb_delete: list of pdbs that don't satisfy length conditions
    :type version: list
    :param pdb_directory: name of directory containing pdb
    :type version: string
    """
    # print(pdb_delete)
    for item in pdb_delete:
        if os.path.exists(f"dscript/{pdb_directory}/{item}.pdb"):
            os.remove(f"dscript/{pdb_directory}/{item}.pdb")


def main():
    [pdb_directory, files, fasta_name, tsv_name, hf_pair] = inputs()

    total = 0
    pdb_delete = []
    errors = []

    for pdb_id in files:
        total += 1
        print(f"Total: {total}")
        print(pdb_id)

        [pdb_file, seqs_long, seqs_short] = get_sequences(
            pdb_directory, pdb_id
        )
        output = check_chains(pdb_id, pdb_file, pdb_delete)
        if output is None:
            continue
        [chains, pdb_delete] = output

        make_fasta(pdb_id, seqs_long, fasta_name)
        make_tsv(pdb_id, chains, tsv_name)

        chains_filtered = filter_chains(chains)
        [
            seq0_long,
            seq0_short,
            chain0,
            seq1_long,
            seq1_short,
            chain1,
        ] = split_sequences(seqs_long, seqs_short, chains_filtered)
        [chain0, chain1] = chain_switch(
            seq0_long, seq0_short, seq1_long, seq1_short, chain0, chain1
        )
        [
            seq0_long_f,
            seq0_short_f,
            seq1_long_f,
            seq1_short_f,
        ] = get_aligned_seqs(seq0_long, seq0_short, seq1_long, seq1_short)

        [D, errors] = calc_dist_matrix(
            pdb_id,
            chain0,
            chain1,
            seq0_long,
            seq1_long,
            seq0_long_f,
            seq0_short_f,
            seq1_long_f,
            seq1_short_f,
            errors,
        )
        hf_pair.create_dataset(
            f"{pdb_id}:{str(chains[0].get_id())}x{pdb_id}:{str(chains[1].get_id())}",
            data=D,
        )

    print(pdb_delete)
    print(errors)
    delete(pdb_delete, pdb_directory)
    delete(errors, pdb_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main()
