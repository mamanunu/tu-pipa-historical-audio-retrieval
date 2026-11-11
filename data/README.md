# Data directory

This repository does not contain historical recordings or audio clips.

To run the analysis on an authorized local copy, provide the two verified genre subdirectories. The program accepts the source-language folder names used by the corpus as well as these romanized aliases:

```text
authorized-audio/
├── Hua Diao/
│   └── *.wav
└── Bei Gong Diao/
    └── *.wav
```

The expected corpus contains 45 Hua Diao recordings and 9 Bei Gong Diao recordings. Source recordings must be mono WAV files. Reproducing the paper's results also requires a private JSON file containing the segment-review decisions. Do not copy local audio, review records, or analysis outputs into this repository.
