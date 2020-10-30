from __future__ import print_function, division

def parse_stream(f, comment=b'#'):

    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)

def parse(f, comment='#'):
    starter = '>'
    empty = ''
    if 'b' in f.mode:
        comment = b'#'
        starter = b'>'
        empty = b''
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

def parse_directory(directory, extension='.seq'):
    names = []
    sequences = []

    for seqPath in os.listdir(directory):
        if seqPath.endswith(extension):
            n, s = parse(open(f"{directory}/{seqPath}","rb"))
            names.append(n[0].decode('utf-8').strip())
            sequences.append(s[0].decode('utf-8').strip())
    return names, sequences

def write(nam, seq, f):
    for n,s in zip(nam,seq):
        f.write('>{}\n'.format(n))
        f.write('{}\n'.format(s))
        
def count_bins(array, bins):
    # Check bins make sense
    lastB = 0
    for b in bins:
            assert b > lastB
            lastB = b
    if bins[0] > min(array) and min(array) < 0:
            bins = [min(array)] + bins
    if bins[-1] < max(array):
            bins.append(max(array))

    binDict = {b: [] for b in bins}

    for i in array:
            for b in range(len(bins)):
                    if i > bins[b]:
                            continue
                    else:
                            binDict[bins[b]].append(i)
                            break

    binLens = {b: len(binDict[b]) for b in bins}

    s = 0
    for b in binDict.keys():
            s += binLens[b]
    assert s == len(array)

    return binLens


