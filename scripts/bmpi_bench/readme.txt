Three scripts for selecting sequences in benchmarking sets:

select_dmel_seqs.py
This script processes FlyBase data to select one D. melanogaster protein sequence per gene and filter by maximum length
Positional arguments:
	1) "FlyBase unique proteins by gene table" listing all genes and corresponding protein names
		(filename, used dmel_unique_protein_isoforms_fb_2025_02.tsv)
	2) Input FASTA of the FlyBase D. Melanogaster proteome (used dmel-all-translation-r6.63.fasta)
	3) Output TSV listing Sequence ID, name, and length for the first occurring protein
		(in the list of names) for each gene in the isoform table, if found in the FASTA
	4) Output FASTA of protein sequences listed in table
	5) Maximum protein length (optional, value used: 1000)
	6) Output FASTA filtered to include only proteins with length <= the max length (optional, with above)

select_endo_seqs.py
This script processes RefSeq data to select sequences with a minimum and maximum length (used for wMel sequences)
Positional arguments:
	1) Input protein FASTA
	2) Minimum length
	3) Maximum length
	4) Output TSV listing the accession and length of all proteins in the input
	5) Output FASTA filtered to only include proteins between the specified minimum and maximum value

sample_pairs.py
This script subsamples pairs of proteins to generate random, sparse sets of pairs
Positional arguments:
	1) File containing a list of proteins, one per line
	2) Fraction of pairs to select, between 0.0 and 1.0
Output to STDOUT: a list of tab-separated pairs, one per line.


Script for monitoring memory usage and notes on collecting data:
mem_monitor.sh measures the available memory every 60 seconds from /proc/meminfo and writes to a log file (the argument)
It runs in the background while various benchmarking tasks are run. For example,

	$ mem_monitor.sh memory.log &
	$ sleep 10 #Allow time to collect initial memory before starting

	#Repeated
	$ echo "Starting run i" >> memory.log
	$ time dscript predict ...
	$ echo "Done with run i" >> memory.log
	$ sleep 180 #Allow time for memory to be freed between tests


Once complete, the positions in the log corresponding to each benchmark are identified
	$ grep -n Start memory.log
	$ grep -n Done memory.log

If, for a particular run, Start in on line S and Done is on line D of the log,
The initial free memory (in KB) is
	$ head -[S-1] memory.log | tail -1 | cut -d " " -f 6
And the minimum free memory is:
	$ head -[D-1] memory.log | tail -n +[S+1] | cut -d " " -f 6 | sort -n | head -1
