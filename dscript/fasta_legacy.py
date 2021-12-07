def parse(f, comment="#"):
    """
    Parse a file in ``.fasta`` format.

    :param f: Input file object
    :type f: _io.TextIOWrapper
    :param comment: Character used for comments
    :type comment: str

    :return: names, sequence
    :rtype: list[str], list[str]
    """
    starter = ">"
    empty = ""
    if "b" in f.mode:
        comment = b"#"
        starter = b">"
        empty = b""
    names = []
    sequences = []
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(starter):
            if name is not None:
                names.append(name)
                sequences.append(empty.join(sequence))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        names.append(name)
        sequences.append(empty.join(sequence))

    return names, sequences


def parse_directory(directory, extension=".seq"):
    """
    Parse all files in a directory ending with ``extension``.

    :param directory: Input directory
    :type directory: str
    :param extension: Extension of all files to read in
    :type extension: str

    :return: names, sequence
    :rtype: list[str], list[str]
    """
    names = []
    sequences = []

    for seqPath in os.listdir(directory):
        if seqPath.endswith(extension):
            n, s = parse(open(f"{directory}/{seqPath}", "rb"))
            names.append(n[0].decode("utf-8").strip())
            sequences.append(s[0].decode("utf-8").strip())
    return names, sequences


def write(nam, seq, f):
    """
    Write a file in ``.fasta`` format.

    :param nam: List of names
    :type nam: list[str]
    :param seq: List of sequences
    :type seq: list[str]
    :param f: Output file object
    :type f: _io.TextIOWrapper
    """
    for n, s in zip(nam, seq):
        f.write(">{}\n".format(n))
        f.write("{}\n".format(s))
