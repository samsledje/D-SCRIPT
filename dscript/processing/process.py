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
        help="Full path of pdb directory containing downloaded pdb files",
    )
    data_grp.add_argument(
        "--pdb_list",
        required=True,
        help="Plain text file containing comma-separated list of PDB IDs in the provided pdb directory.",
    )
    data_grp.add_argument(
        "--filter_chain_minlen",
        required=True,
        help="Filtering step: enter desired minimum length of chain",
    )
    data_grp.add_argument(
        "--filter_chain_maxlen",
        required=True,
        help="Filtering step: enter desired maximum length of chain",
    )
    data_grp.add_argument(
        "--h5_name",
        required=True,
        help="Full path of output H5 (for contact maps)",
    )
    data_grp.add_argument(
        "--fasta",
        required=True,
        help="Full path of output fasta (for sequences)",
    )
    data_grp.add_argument(
        "--tsv",
        required=True,
        help="Full path of output tsv (for PPIs)",
    )

    return parser


def get_pdb_list(pdb_text):
    pdb_list = []
    with open(f"{pdb_text}", "r") as pdb_f:
        pdb_list = pdb_f.read().split(", ")
    return pdb_list


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
    pdb_file = f"{pdb_directory}/{pdb_id}.pdb"
    seqres_recs = list(SeqIO.parse(pdb_file, "pdb-seqres"))
    atoms_recs = list(SeqIO.parse(pdb_file, "pdb-atom"))
    seqs_long = seqres_recs[:2]
    seqs_short = atoms_recs[:2]
    return [pdb_file, seqs_long, seqs_short]


def get_filtered_chains(
    pdb_id, pdb_file, chain_minlen, chain_maxlen, chain_error, chain_few
):
    """
    Gets atom and seqres sequences from the pdb file.

    :param pdb_id: name of pdb file
    :type version: string
    :param pdb_file: path to pdb file
    :type version: string
    :param chain_minlen: minimum length of chain filtered out
    :type version: int
    :param chain_maxlen: maximum length of chain filtered out
    :type version: int
    :param chain_error: list of pdbs that don't satisfy length conditions
    :type version: list
    :param chain_few: list of pdbs that have more than two chains
    :type version: list
    :return: list containing a list of filtered chains and two list of filtered out pdbs
    :rtype: list
    """
    structure = PDB.PDBParser().get_structure(pdb_id, pdb_file)
    chains = list(structure.get_chains())
    if len(chains) > 2:
        chain_few.append(pdb_id)
        return None
    chains = list(structure.get_chains())[:2]
    if (
        len(chains[0]) > chain_maxlen
        or len(chains[1]) > chain_maxlen
        or len(chains[0]) < chain_minlen
        or len(chains[1]) < chain_minlen
    ):
        chain_error.append(pdb_id)
        return None
    return [chains, chain_error, chain_few]


def make_fasta_and_tsv(tsv_name, fasta_name, valid_pdb, pdb_directory):
    """
    Creates/appends protein chains to a fasta and a tsv file.

    :param tsv_name: path of tsv file
    :type version: string
    :param fasta_name: path of fasta file
    :type version: string
    :param valid_pdb: list of valid pdbs that passed filtering
    :type version: list
    :param pdb_directory: name of pdb-containing directory
    :type version: string
    """
    with open(f"{tsv_name}", "w+") as tsv_f, open(
        f"{fasta_name}", "w+"
    ) as fasta_f:
        for pdb_id in valid_pdb:
            pdb_file = f"{pdb_directory}/{pdb_id}.pdb"
            seqres_recs = list(SeqIO.parse(pdb_file, "pdb-seqres"))
            for record in seqres_recs[:2]:
                fasta_f.write(record.format("fasta-2line"))

            tsv_writer = csv.writer(tsv_f, delimiter="\t")
            structure = PDB.PDBParser().get_structure(pdb_id, pdb_file)
            chains = list(structure.get_chains())[:2]
            prot1 = f"{pdb_id.upper()}:{str(chains[0].get_id()).upper()}"
            prot2 = f"{pdb_id.upper()}:{str(chains[1].get_id()).upper()}"
            tsv_writer.writerow([prot1, prot2, "1"])


def remove_CA_from_chains(chain_list):
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


def main(args):
    pdb_directory = args.pdb_directory
    h5_name = args.h5_name
    fasta_name = args.fasta
    tsv_name = args.tsv
    pdb_text = args.pdb_list
    pdb_list = get_pdb_list(pdb_text)
    chain_minlen = int(args.filter_chain_minlen)
    chain_maxlen = int(args.filter_chain_maxlen)

    with h5py.File(f"{h5_name}", "w") as hf_pair:
        total = 0
        valid_pdb = []
        chain_error = []
        chain_few = []
        errors = []
        seq_error = []

        for pdb_id in pdb_list:
            total += 1
            print(f"Total: {total}")
            print(pdb_id)

            [pdb_file, seqs_long, seqs_short] = get_sequences(
                pdb_directory, pdb_id
            )
            if len(seqs_short) < 2 or len(seqs_long) < 2:
                seq_error.append(pdb_id)
                continue
            output = get_filtered_chains(
                pdb_id,
                pdb_file,
                chain_minlen,
                chain_maxlen,
                chain_error,
                chain_few,
            )
            if output is None:
                continue
            [chains, chain_error, chain_few] = output

            valid_pdb.append(pdb_id)

            chains_filtered = remove_CA_from_chains(chains)
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

        print(chain_error)
        print(chain_few)
        print(errors)
        print(seq_error)

    make_fasta_and_tsv(tsv_name, fasta_name, valid_pdb, pdb_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    print(parser.parse_args())
    main(parser.parse_args())
