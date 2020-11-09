from dscript.lm_embed import embed_from_fasta
import sys

try:
    inPath = sys.argv[1]
    outPath = sys.argv[2]
    device = int(sys.argv[3])
except IndexError:
    print('usage: python embed.py [input fasta] [output h5] [device]')
    sys.exit(1)
    
embed_from_fasta(inPath,outPath,device)