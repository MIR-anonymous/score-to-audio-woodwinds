# Git repository accompanying submission 277: Score-to-Audio Alignment for Solo Woodwind Instrument Recordings.

## Data

### Recordings

- [Recordings metadata](recordings.csv) : Metadata including Youtube playlist ID

### Annotations files

- [annotations A1](annotations/A1/) : 8 performers and 12 fantasias for each
- [annotations A2](annotations/A2/) : fantasia No. 1 for 8 performers

### Scores

- [pdf](scores/pdf/)
- [unfolded scores](scores/unfolded-scores/)

### MPE outputs

- [output files from Neural MPE model](npz/)

### Unshifted and representation-shifted alignments

- [Chroma-based](ouputs/chroma/)
- [Spectral Template-based](outputs/st/)
- [Neural MPE-based](ouputs/neural/)

## Code

- [Requirements](requirements.txt): List of Python packages and versions used for the alignments  
- [Alignment](alignments/) : Scripts for reproducing alignments (un-shifted and shifted) for the 3 audio feature representations  
- [MPE features](mpe-interference/) : model and code for reproducing **Neural MPE** outputs (requires _PyTorch_ + _PyTorchLightning_)
