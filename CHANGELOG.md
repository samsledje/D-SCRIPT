# To Do
- Full logging system (issue #5)
- Add multi-gpu support (issue #6)
- Use multiple workers to load embeddings / support for loading embeddings on the fly to reduce memory usage (issue #8/11)
- Add convenience function to generate candidates - all pairs from a list / cartesian produt of multiple lists

# v0

## v0.1

### v0.1.5: Active
- Updated package level imports 
- Updated documentation

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
