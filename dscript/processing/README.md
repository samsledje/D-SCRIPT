# PDB Contact Map Processing
These files provide the necessary code for generating 2D contact maps from (heterodimer) protein pdb files from the RCSB PDB using a contact-based approach, representing structures of complexes by pairwise distances between residues of each protein.
Processing also allows for filtering by several parameters, including protein length; number of protein chains in the complex; and distance, discontinuity, and interaction thresholds. Additionally, the pipeline handles several sources of noise, such as non-standard residues and residues with missing 3D coordinates.
The processing files generate an H5 dataset of contact maps in compact form. Additionally, the pipeline extracts protein sequences and generates a dataset of protein interactions in standard formats (FASTA, TSV) for efficient processing by machine learning models.

## Running Code
1. Make directory for storing data
    `mkdir [directory]`
2. Download and unzip PDB files for complexes (Input file: text file containing comma-separated PDB IDs)
    `bash bin/download_contact_maps.sh -f [input file] -o [directory] -p; gunzip [directory]/*.gz`
3. Run format.py to create a plain text file [pdb_file] with each line containing a full path to a PDB file
    (ex. [directory1]/[15C8.pdb]
         [directory2]/[25C8.pdb]
         ...)
3. Run processing code, passing in the desired paths of output files, the plain text file of pdbs, and filtering parameters
    `dscript process --pdb_files [pdb_file] --filter_chain_minlen [min_len] --filter_chain_maxlen [max_len] --h5_name [dataset path] --fasta [fasta path] --tsv [tsv path] --distance_threshold [x angstroms] --discontinuity_threshold [percentage threshold] --interaction_threshold [percentage threshold]`