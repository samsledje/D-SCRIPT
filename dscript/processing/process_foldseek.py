from Bio import SeqIO
from Bio import SeqRecord
from Bio import Seq
from Bio import PDB
from Bio import pairwise2
import h5py
import numpy as np
import pandas as pd
import csv
import argparse
from datetime import datetime
import time
from pathlib import Path
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

MAX_D = 25


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """
    data_grp = parser.add_argument_group("Data Processing files")
    data_grp.add_argument(
        "--pdb_fasta",
        required=True,
        help="fasta of pdb aa fseek",
    )
    data_grp.add_argument(
        "--fseek_fasta",
        required=True,
        help="Full path of unaligned foldseek fasta (for sequences)",
    )
    data_grp.add_argument(
        "--cmap_fasta",
        required=True,
        help="Full path of newly generated contact map sequences",
    )
    data_grp.add_argument(
        "--fasta",
        required=True,
        help="Full path of output fasta (for sequences)",
    )
    return parser


def get_pdb_list(pdb_file):
    """
    Returns a list of pdb file paths given a text file containing pdb paths.

    :param pdb_files: Name of txt file containing pdb full paths.
    :type version: string
    :return: list of pdb files (full path)
    :rtype: list
    """
    pdb_list = []
    with open(f"{pdb_file}", "r") as pdb_f:
        pdb_list = pdb_f.read().split("\n")
    if "" in pdb_list:
        pdb_list.remove("")
    return pdb_list


def get_sequences_from_chains(chains, pair):
    """
    Returns a list of Atom Sequences chains

    :param chains: list of all chains in pdb
    :type version: list
    :param pair: tuple pair of indices taken from chains
    :type version: string
    :return: list of atom sequences
    :rtype: list
    """
    substitutions = {
        "2AS": "ASP",
        "3AH": "HIS",
        "5HP": "GLU",
        "ACL": "ARG",
        "AGM": "ARG",
        "AIB": "ALA",
        "ALM": "ALA",
        "ALO": "THR",
        "ALY": "LYS",
        "ARM": "ARG",
        "ASA": "ASP",
        "ASB": "ASP",
        "ASK": "ASP",
        "ASL": "ASP",
        "ASQ": "ASP",
        "AYA": "ALA",
        "BCS": "CYS",
        "BHD": "ASP",
        "BMT": "THR",
        "BNN": "ALA",
        "BUC": "CYS",
        "BUG": "LEU",
        "C5C": "CYS",
        "C6C": "CYS",
        "CAS": "CYS",
        "CCS": "CYS",
        "CEA": "CYS",
        "CGU": "GLU",
        "CHG": "ALA",
        "CLE": "LEU",
        "CME": "CYS",
        "CSD": "ALA",
        "CSO": "CYS",
        "CSP": "CYS",
        "CSS": "CYS",
        "CSW": "CYS",
        "CSX": "CYS",
        "CXM": "MET",
        "CY1": "CYS",
        "CY3": "CYS",
        "CYG": "CYS",
        "CYM": "CYS",
        "CYQ": "CYS",
        "DAH": "PHE",
        "DAL": "ALA",
        "DAR": "ARG",
        "DAS": "ASP",
        "DCY": "CYS",
        "DGL": "GLU",
        "DGN": "GLN",
        "DHA": "ALA",
        "DHI": "HIS",
        "DIL": "ILE",
        "DIV": "VAL",
        "DLE": "LEU",
        "DLY": "LYS",
        "DNP": "ALA",
        "DPN": "PHE",
        "DPR": "PRO",
        "DSN": "SER",
        "DSP": "ASP",
        "DTH": "THR",
        "DTR": "TRP",
        "DTY": "TYR",
        "DVA": "VAL",
        "EFC": "CYS",
        "FLA": "ALA",
        "FME": "MET",
        "GGL": "GLU",
        "GL3": "GLY",
        "GLZ": "GLY",
        "GMA": "GLU",
        "GSC": "GLY",
        "HAC": "ALA",
        "HAR": "ARG",
        "HIC": "HIS",
        "HIP": "HIS",
        "HMR": "ARG",
        "HPQ": "PHE",
        "HTR": "TRP",
        "HYP": "PRO",
        "IAS": "ASP",
        "IIL": "ILE",
        "IYR": "TYR",
        "KCX": "LYS",
        "LLP": "LYS",
        "LLY": "LYS",
        "LTR": "TRP",
        "LYM": "LYS",
        "LYZ": "LYS",
        "MAA": "ALA",
        "MEN": "ASN",
        "MHS": "HIS",
        "MIS": "SER",
        "MLE": "LEU",
        "MPQ": "GLY",
        "MSA": "GLY",
        "MSE": "MET",
        "MVA": "VAL",
        "NEM": "HIS",
        "NEP": "HIS",
        "NLE": "LEU",
        "NLN": "LEU",
        "NLP": "LEU",
        "NMC": "GLY",
        "OAS": "SER",
        "OCS": "CYS",
        "OMT": "MET",
        "PAQ": "TYR",
        "PCA": "GLU",
        "PEC": "CYS",
        "PHI": "PHE",
        "PHL": "PHE",
        "PR3": "CYS",
        "PRR": "ALA",
        "PTR": "TYR",
        "PYX": "CYS",
        "SAC": "SER",
        "SAR": "GLY",
        "SCH": "CYS",
        "SCS": "CYS",
        "SCY": "CYS",
        "SEL": "SER",
        "SEP": "SER",
        "SET": "SER",
        "SHC": "CYS",
        "SHR": "LYS",
        "SMC": "CYS",
        "SOC": "CYS",
        "STY": "TYR",
        "SVA": "SER",
        "TIH": "ALA",
        "TPL": "TRP",
        "TPO": "THR",
        "TPQ": "ALA",
        "TRG": "LYS",
        "TRO": "TRP",
        "TYB": "TYR",
        "TYI": "TYR",
        "TYQ": "TYR",
        "TYS": "TYR",
        "TYY": "TYR",
    }
    residues = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    records = []
    # chains = chains[:2]
    chains_two = []
    chains_two.append(chains[pair[0]])
    chains_two.append(chains[pair[1]])

    for chain in chains_two:
        chain_string = ""
        for residue in chain:
            if residue.has_id("CA"):
                if residue.get_resname() in substitutions.keys():
                    chain_string += "" + PDB.Polypeptide.three_to_one(
                        substitutions[residue.get_resname()]
                    )
                elif residue.get_resname() == "CA":
                    None
                elif residue.get_resname() not in residues:
                    # print(residue.get_resname())
                    chain_invalid = chain
                    chain_string = None
                    break
                else:
                    chain_string += "" + PDB.Polypeptide.three_to_one(
                        residue.get_resname()
                    )
        if chain_string is None:
            return [records, chain_invalid]
        chain_seq = Seq.Seq(chain_string)
        chain_record = SeqRecord.SeqRecord(
            chain_seq, name=chain.id, id=chain.id
        )
        records.append(chain_record)
    return records


def get_sequences(pdb, chains, pair):
    """
    Gets atom and seqres sequences from the pdb file.

    :param pdb: full path of pdb file
    :type version: string
    :param chains: list of all chains in pdb
    :type version: list
    :param pair: tuple pair of indices taken from chains
    :type version: string
    :return: list containing sequence from seq-res and sequence from atom
    :rtype: list
    """
    atoms_recs = get_sequences_from_chains(chains, pair)
    if not atoms_recs[0] or len(atoms_recs[0]) == 1:
        return [None, atoms_recs[1]]
    seqs_short = atoms_recs
    seqres_recs = list(SeqIO.parse(pdb, "pdb-seqres"))
    seqs_long = []
    seqs_long.append(seqres_recs[pair[0]])
    seqs_long.append(seqres_recs[pair[1]])
    return [seqs_long, seqs_short]


def check_sequences_valid(seqs_long, seqs_short):
    """
    Checks if sequences are both valid amino acid chains, not "XX...XX".

    :param seqs_long: list of seq-res sequences for chains 0 and 1
    :type version: string
    :param seqs_short: list of atom sequences for chains 0 and 1
    :type version: string
    :return: whether both sequences exist and are valid
    :rtype: Boolean
    """
    # print((seqs_long[0].seq, seqs_long[1].seq))
    # print((seqs_short[0].seq, seqs_short[1].seq))
    # print("\n")
    return (
        len(seqs_short) < 2
        or len(seqs_long) < 2
        or len(str(seqs_long[0].seq)) == 0
        or len(str(seqs_long[1].seq)) == 0
        or (
            str(seqs_long[0].seq)
            == len(str(seqs_long[0].seq)) * str(seqs_long[0].seq)[0]
        )
        or (
            str(seqs_long[1].seq)
            == len(str(seqs_long[1].seq)) * str(seqs_long[1].seq)[0]
        )
    )


def get_chains_prelim_filtering(
    chains, pdb_id, chain_minlen, chain_maxlen, chain_error, pair
):
    """
    Gets atom and seqres sequences from the pdb file.

    :param chains: list of chains contained in pdb file
    :type version: list
    :param pdb_id: name of pdb file
    :type version: string
    :param chain_minlen: minimum length of chain filtered out
    :type version: int
    :param chain_maxlen: maximum length of chain filtered out
    :type version: int
    :param chain_error: list of pdbs that don't satisfy length conditions
    :type version: list
    :param pair: tuple pair of indices taken from chains
    :type version: string
    :return: a list of filtered chains and two lists of pdbs that don't satisfy filtering conditions
    :rtype: list
    """
    chains_two = []
    chains_two.append(chains[pair[0]])
    chains_two.append(chains[pair[1]])
    if len(chains_two[0]) < chain_minlen or len(chains_two[0]) > chain_maxlen:
        if f"{pdb_id}:{str(chains_two[0].get_id())}" not in chain_error:
            chain_error.append(f"{pdb_id}:{str(chains_two[0].get_id())}")
            print(
                f"REMOVED Chain {str(chains_two[0].get_id())} - Length Constraints"
            )
        return None
    if len(chains_two[1]) < chain_minlen or len(chains_two[1]) > chain_maxlen:
        if f"{pdb_id}:{str(chains_two[1].get_id())}" not in chain_error:
            chain_error.append(f"{pdb_id}:{str(chains_two[1].get_id())}")
            print(
                f"REMOVED Chain {str(chains_two[1].get_id())} - Length Constraints"
            )
        return None
    return [chains_two, chain_error]


def make_fasta_and_tsv(
    tsv_name, fasta_name, valid_pdb, pdbs, chain_error, invalid_resname
):
    """
    Takes in a list of valid pdb file paths and creates an output fasta and tsv file.

    :param tsv_name: path of tsv file
    :type version: string
    :param fasta_name: path of fasta file
    :type version: string
    :param valid_pdb: list of valid pdbs that passed filtering
    :type version: list
    :param pdbs: list of all valid pdb
    :type version: list
    :param chain_error: list of all pdb chains that have invalid lengths
    :type version: list
    :param invalid_resname: list of all pdbs that have invalid residues
    :type version: list
    """
    with open(f"{tsv_name}", "w+") as tsv_f, open(
        f"{fasta_name}", "w+"
    ) as fasta_f:
        for pdb in pdbs:
            sequences = list(SeqIO.parse(pdb, "pdb-seqres"))
            for record in sequences:
                # print(record)
                if (
                    record.id not in chain_error
                    and record.id not in invalid_resname
                    and record.name != "<unknown name>"
                    and len(record.seq) > 0
                    and record.seq != str(record.seq[0]) * len(str(record.seq))
                ):
                    fasta_f.write(record.format("fasta-2line"))
        for pdb_pair in valid_pdb.keys():
            tsv_writer = csv.writer(tsv_f, delimiter="\t")
            tsv_writer.writerow([pdb_pair[0:6], pdb_pair[7:13], "1"])


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
    :param chains_filtered: list of filtered chains
    :type version: list
    :return: list containing: seq-res sequence for chain0, atom sequence for chain0, chain0, seq-res sequence for chain1, atom sequence for chain1, chain1
    :rtype: list
    """
    seq0_long = seqs_long[0].seq
    seq0_short = seqs_short[0].seq
    chain0 = chains_filtered[0]
    seq1_long = seqs_long[1].seq
    seq1_short = seqs_short[1].seq
    chain1 = chains_filtered[1]
    return [seq0_long, seq0_short, chain0, seq1_long, seq1_short, chain1]


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


def residue_distance(res0, res1, max_d):
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


def calc_foldseek_align(
    seq_fasta,
    seq_pdb,
    seq_fseek,
    seq_fasta_f,
    seq_pdb_f,
):
    # try-catch
    try:
        assert len(seq_pdb) == len(seq_fseek)
    except:
        print(f"{len(seq_pdb)}, {len(seq_fseek)}")

    fseek_iter = iter(seq_fseek)
    fseek_align_char = []
    for (res0_fasta, res0_pdb) in zip(seq_fasta_f, seq_pdb_f):
        if res0_fasta == "-":
            continue
        elif res0_pdb == "-":
            fseek_align_char.append("X")
        else:
            fseek_align_char.append(next(fseek_iter))
    return "".join(fseek_align_char)


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
    dist_thresh,
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
    :param dist_thresh: argument for distance threshold
    :type version: float
    :return: generated distance matrix between two chains
    :rtype: Numpy matrix
    """
    D = np.zeros((len(seq0_long), len(seq1_long)))
    D = D + 25
    ch0_it = iter(chain0)
    x = -1
    y = 0
    for (i, (res0L, res0S)) in enumerate(zip(seq0_long_f, seq0_short_f)):

        if res0S == "-":
            x += 1
            continue
        elif res0L == "-":
            res0 = next(ch0_it)
            continue
        else:
            x += 1
            res0 = next(ch0_it)

        ch1_it = iter(chain1)
        for (j, (res1L, res1S)) in enumerate(zip(seq1_long_f, seq1_short_f)):
            if res1S == "-":
                y += 1
                continue
            elif res1L == "-":
                res1 = next(ch1_it)
                continue
            else:
                res1 = next(ch1_it)
                D[x, y] = residue_distance(res0, res1, dist_thresh)
                y += 1
        y = 0
    df = pd.DataFrame(D, index=list(seq0_long), columns=list(seq1_long))
    # print(df)
    return D


def log(m, file=None, timestamped=True, print_also=False):
    """
    Outputs a log.

    :param m: information to be logged
    :type version: Object
    :param file: file name for log to be written to
    :type version: string
    :param timestamped: whether or not log will include timestamp
    :type version: boolean
    :param print_also: whether or not log is printed to standard output as well as written to file
    :type version: boolean
    """
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "
    log_string = f"{curr_time if timestamped else ''}{m}"
    if file is None:
        print(log_string)
    else:
        print(log_string, file=file)
        if print_also:
            print(log_string)
        file.flush()


def main(args):
    # input fasta = new_proteins (id, seq) seqio reader
    # get pdb structure version of sequence
    # input foldseek fasta = foldseek seq
    # alignment b/w fasta and pdb, then aligned seq b/w foldseek seqs
    #
    start = time.perf_counter()
    cmap_fasta = args.cmap_fasta
    fseek_fasta = args.fseek_fasta
    pdb_fasta = args.pdb_fasta
    output_fasta = args.fasta

    cmap_dict = {}
    for record in SeqIO.parse(cmap_fasta, "fasta"):
        pdb = record.id.split(":")[0]
        cmap_dict[record.id] = {"cmap": str(record.seq)}
        # if len(cmap_dict) > 50:
        #     break

    pdb_dict = {
        record.id: str(record.seq)
        for record in SeqIO.parse(pdb_fasta, "fasta")
    }
    for pdb_chain in tqdm(cmap_dict.keys(), total=len(cmap_dict)):
        (pdb, chain) = pdb_chain.split(":")
        f_key = f"{pdb.lower()}.cif.gz_{chain.upper()}"
        try:
            cmap_dict[pdb_chain]["pdb"] = pdb_dict[f_key]
        except KeyError:
            cmap_dict[pdb_chain]["pdb"] = "X" * len(
                cmap_dict[pdb_chain]["cmap"]
            )

    fseek_dict = {
        record.id: str(record.seq)
        for record in SeqIO.parse(fseek_fasta, "fasta")
    }

    counter = 0
    for pdb_chain in tqdm(cmap_dict.keys(), total=len(cmap_dict)):
        (pdb, chain) = pdb_chain.split(":")
        f_key = f"{pdb.lower()}.cif.gz_{chain.upper()}"
        try:
            cmap_dict[pdb_chain]["fseek"] = fseek_dict[f_key]
        except KeyError:
            counter += 1
            cmap_dict[pdb_chain]["fseek"] = "X" * len(
                cmap_dict[pdb_chain]["cmap"]
            )

    # print(cmap_dict)
    print(f"Counting # times fseek was missing: {counter}")
    print(f"Total # keys: {len(cmap_dict)}")

    fseek_aligned = {}

    for pdb_chain in tqdm(cmap_dict.keys(), total=len(cmap_dict)):
        seq_fasta = cmap_dict[pdb_chain]["cmap"]
        seq_pdb = cmap_dict[pdb_chain]["pdb"]
        seq_fseek = cmap_dict[pdb_chain]["fseek"]

        seqs_align_f = pairwise2.align.globalxx(seq_fasta, seq_pdb)
        seq_fasta_f = seqs_align_f[0].seqA
        seq_pdb_f = seqs_align_f[0].seqB

        seq_fseek_f = calc_foldseek_align(
            seq_fasta,
            seq_pdb,
            seq_fseek,
            seq_fasta_f,
            seq_pdb_f,
        )

        fseek_aligned[pdb_chain] = seq_fseek_f

    with open(output_fasta, "w+") as file:
        for pdb_chain, seq in fseek_aligned.items():
            file.write(f">{pdb_chain}\n{seq}\n")

    ########################
    with h5py.File(f"{h5_name}", "w") as hf_pair:
        total = 0
        valid_pdb = {}
        pdbs = []
        chain_error = []
        seq_error = []
        invalid_resname = []
        chainlen_unsatisfied = []
        unknown_name = []

        for pdb in pdb_list:
            total += 1
            print(f"Total: {total}")
            print(pdb)
            pdb_id = Path(pdb).stem

            structure = PDB.PDBParser().get_structure(pdb_id, pdb)
            sequences = list(SeqIO.parse(pdb, "pdb-seqres"))
            flag = False
            for item in sequences:
                if item.name == "<unknown name>":
                    flag = True
                    if pdb_id not in unknown_name:
                        unknown_name.append(pdb_id)
            if flag:
                continue
            chains = list(structure.get_chains())
            if len(chains) != len(sequences):
                seq_error.append(pdb_id)
                continue
            # print(chains)
            if args.filter_number_of_chains is not None:
                if len(chains) not in chain_lengths_allowed:
                    chainlen_unsatisfied.append(pdb_id)
                    continue
            chains.sort(key=lambda chain: chain.id)
            chain_pairing = [item for item in range(0, len(chains))]
            pair = [
                (a, b)
                for idx, a in enumerate(chain_pairing)
                for b in chain_pairing[idx + 1 :]
            ]
            # print(pair)
            for item in pair:
                seqences_pair = get_sequences(pdb, chains, item)
                if seqences_pair[0] is None:
                    if (
                        f"{pdb_id}:{str(seqences_pair[1].id)}"
                        not in invalid_resname
                    ):
                        invalid_resname.append(
                            f"{pdb_id}:{str(seqences_pair[1].id)}"
                        )
                        print("REMOVED - Invalid Residue Name")
                    continue
                [seqs_long, seqs_short] = seqences_pair

                seq_verify = check_sequences_valid(seqs_short, seqs_long)
                if seq_verify is True:
                    if pdb_id not in seq_error:
                        seq_error.append(pdb_id)
                        print("REMOVED - <2 Sequences")
                    continue

                output = get_chains_prelim_filtering(
                    chains,
                    pdb_id,
                    chain_minlen,
                    chain_maxlen,
                    chain_error,
                    item,
                )
                if output is None:
                    continue
                [chains_two, chain_error] = output

                # print(chains_two)
                # print(seqs_long)
                # print(seqs_short)

                if pdb not in pdbs:
                    pdbs.append(pdb)

                chains_filtered = remove_CA_from_chains(chains_two)
                [
                    seq0_long,
                    seq0_short,
                    chain0,
                    seq1_long,
                    seq1_short,
                    chain1,
                ] = split_sequences(seqs_long, seqs_short, chains_filtered)

                [
                    seq0_long_f,
                    seq0_short_f,
                    seq1_long_f,
                    seq1_short_f,
                ] = get_aligned_seqs(
                    seq0_long, seq0_short, seq1_long, seq1_short
                )
                seq_fseek_f = calc_foldseek_align(
                    seq_fasta,
                    seq_pdb,
                    seq_fseek,
                    seq_fasta_f,
                    seq_pdb_f,
                )

    make_fasta_and_tsv(
        tsv_name, fasta_name, valid_pdb, pdbs, chain_error, invalid_resname
    )
    log(f"PDB chains that <{chain_minlen} or >{chain_maxlen}: {chain_error}")
    log(
        f"PDBs with an invalid # of sequences (<2 or != # chains): {seq_error}"
    )
    log(
        f"PDB chains with invalid residue names (e.g. MSE, PLM): {invalid_resname}"
    )
    log(
        f"PDBs with chain lengths that don't satisfy user-entered constraints: {chainlen_unsatisfied}"
    )
    log(f"PDBs with unknown names/descriptions: {unknown_name}")
    end = time.perf_counter()
    print(f"Elapsed {(end-start)/60} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    print(parser.parse_args())
    main(parser.parse_args())
