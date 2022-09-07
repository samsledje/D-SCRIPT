# PDB Contact Map Processing
These files provide the necessary code for generating 2D contact maps from (heterodimer) protein pdb files from the RCSB PDB. The data processing outputs include generating an h5 dataset for contact maps, an output fasta file for protein sequences, and a tsv file for protein interactions.

## Running Code
1. Make directory for storing data
    `mkdir dscript/pdbs_large`
2. Download and unzip PDB files for complexes
    `bash bin/download_contact_maps.sh -f dscript/processing/pdb_ids.txt -o dscript/pdbs_large -p; gunzip dscript/pdbs_large/*.gz`
3. Generate contact map dataset, output fasta, and output tsv
    `python dscript/processing/process.py --pdb_directory pdbs_large --h5_name output_l --fasta proteins_l --tsv cmap_l`