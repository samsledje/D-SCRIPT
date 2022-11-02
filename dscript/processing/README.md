# PDB Contact Map Processing
These files provide the necessary code for generating 2D contact maps from (heterodimer) protein pdb files from the RCSB PDB. The data processing outputs include generating an h5 dataset for contact maps, an output fasta file for protein sequences, and a tsv file for protein interactions.

## Running Code
1. Make directory for storing data
    `mkdir [directory]`
2. Download and unzip PDB files for complexes (Input file: text file containing comma-separated PDB IDs)
    `bash bin/download_contact_maps.sh -f [input file] -o [directory] -p; gunzip [directory]/*.gz`
3. Create a plain text file [pdb_files] with each line containing a full path to a PDB file
    (ex. [directory1]/[15C8.pdb]
         [directory2]/[25C8.pdb]
         ...)
3. Run processing code (provide full file paths for [pdb_files, h5_name, fasta, and tsv])
    `python dscript/processing/process.py --pdb_files [pdb_files] --filter_chain_minlen [min_len] --filter_chain_maxlen [max_len] --h5_name [dataset] --fasta [fasta] --tsv [tsv]`