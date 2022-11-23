# To Do
- Expand test suite to include function unittests (maximize coverage)
- Add multi-gpu support (issue #6)
- support for loading embeddings on the fly to reduce memory usage (issue #11)

# v0

## v0.2

### v0.2.2
- Resolve #35 to use `require_dataset` -- can now add multiple .fasta files to the same h5 file
- Update pretrained API and docs to include Topsy-Turvy
- Add retry decorator to get_pretrained if download fails
- Add ability to set a random seed for training

### v0.2.1: 2022-06-28 -- Bug fixes
- Add biopython to setup.py

### v0.2.0: 2022-06-24 -- Integration of Topsy Turvy

- Integrate Topsy-Turvy to allow for top-down supervision
- Use utils.log function across all commands
- Speed up loading embeddings into memory using parallel processing
- Update fasta parse and write to use BioPython SeqIO (better error checking)
- More comprehensive test suite for main commands

## v0.1

### v0.1.9: 2022-03-07 -- Minor bug fixes

- Updated model loading on new version to handle re-named parameters
- Updated cpu-only loading during prediction with map_location

### v0.1.8: 2022-02-08 -- Finalized training bug fix

- Resolve #24 by fixing training
- Can now run `dscript train --train data/pairs/human_train.tsv --test data/pairs/human_test.tsv --embedding /afs/csail/u/s/samsl/Work/databases/STRING/homo.sapiens/human_nonRed.h5 --output [output] --save-prefix [prefix] --device 0` to replicate paper results
- Updated code formatting with black and pre-commit

### v0.1.7-dev9: 2022-02-07 -- Update training code to be clean again

- Following previous update, addresses #24 by fixing model training while maintaining preferred API and command line usage

### v0.1.7-dev1: 2021-12-16 -- Major overhaul in training

- Fixed significant bug in how training was run by reverting to older code
- Should address issue #24: unable to replicate paper results
- To do: code cleaning to bring up to formatting standards while maintaining performance

### v0.1.6: 2021-09-06 -- Bug Fix - Augmentation and proper defaults
- Augmentation fix in v0.1.5 was bugged still and would throw an error, now resets index
- Change `--use-w` and `--augment` to `--no-w` and `--no-augment` with store false

### v0.1.5: 2021-06-23 -- Bug Fix - Augment and Documentation
- Updated package level imports
- Updated documentation
- Fixed issue #13: improper augmentation of data
- Fixed issue #12: overwrites cmap data sets if they already exist

### v0.1.4: 2021-03-05 -- Bug Fix - Typo in `ContactModule.forward()`
- Fixed issue #7: bug which would crash contact module if called directly

### v0.1.3: 2021-02-17 -- Bug Fix - Pairs too large for GPU
- Fixed issues #3, #4
- Basic logging system implemented to report skipped pairs
- Fixed wrong variable name in loading from sequence file
- Updated documentation

### v0.1.2: 2020-11-30 -- Bug Fix - Eval Mode
- Model should be put into `eval()` mode before prediction or evaluation, and when new models are downloaded - this makes the output deterministic by disabling dropout layers

### v0.1.1: 2020-11-18 -- First Beta Release
