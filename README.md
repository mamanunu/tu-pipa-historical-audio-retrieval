# Computational Organization and Machine-Listening Retrieval for Historical Tu Pipa Field Recordings from Northwestern Sichuan

This repository provides the analysis code, method parameters, public aggregate results, and a verification script for the accompanying paper. The study examines 54 restricted-access historical field recordings, creates 270 fixed-duration segments, compares three short-time acoustic representations, and evaluates nearest-neighbor retrieval against a random baseline and under method-sensitivity checks.

The genre labels are romanized as **Hua Diao** and **Bei Gong Diao** to preserve the source taxonomy; no literal English translation is intended.

## Repository contents

- `src/build_corpus_analysis.py`: runs the full analysis on an authorized local copy of the recordings.
- `src/verify_public_outputs.py`: checks that the public aggregate results are complete and match the paper's reported values.
- `docs/`: method parameters, public field definitions, and data-availability information.
- `results/`: aggregate statistics and figures used in the paper.
- `data/README.md`: describes the unavailable data and the required local input structure.
- `config/config.example.json`: an example of local parameter settings; the program reads the corresponding values from command-line options.

## Data availability

The repository does not include the source recordings, four-second audio clips, performer-level details, segment time indexes, or segment-level acoustic features. These materials are restricted by recording permissions and performer rights. The code license does not apply to the recordings or to unpublished derived data. See `docs/data_availability.md` for details.

## Environment

- Python 3.12
- NumPy 2.3.5
- Pillow 13.0.0
- FFmpeg 8.1.1, including both `ffmpeg` and `ffprobe`

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Verify the public results

```bash
python src/verify_public_outputs.py
```

The script checks the row counts, genre labels, and key reported values in the public aggregate tables. It does not recompute audio features, nearest-neighbor rankings, or confidence intervals. A full rerun requires authorized local recordings and the private manual-review records.

## Run on authorized material

The input directory must contain the two verified genre folders of mono WAV files. The script accepts the source-language folder names used by the corpus as well as the romanized aliases `Hua Diao/` and `Bei Gong Diao/`; it does not rename or modify the source material. To reproduce the paper's analysis, also provide the complete private JSON file containing the segment-review decisions. The `--manual-cases` file is optional.

```bash
python src/build_corpus_analysis.py \
  --source-root /path/to/authorized/audio \
  --output-root /path/to/private/analysis \
  --manual-review /path/to/private/manual_review_status.json \
  --manual-cases /path/to/private/manual_case_interpretations.json \
  --collector "collector metadata" \
  --period "recording period" \
  --location "recording region" \
  --source-medium "source medium" \
  --permission-status "local access conditions"
```

The program does not export four-second WAV clips by default. Add `--write-derived-audio` only when the permissions allow it and the clips are necessary. `--accept-unreviewed` is for demonstrating the workflow on a corpus with the same 54-recording, 45/9 structure; it does not reproduce the paper's reviewed analysis. This paper-specific pipeline checks the corpus size and category balance and is not a general-purpose training framework.

## Scope

The code implements exploratory nearest-neighbor retrieval within this corpus, not an automatic genre classifier trained and tested on unseen recordings. The public results describe relationships under the paper's 54-recording corpus, fixed segmentation rules, and three feature sets.

## Citation

When citing this repository, use the information in `CITATION.cff` and cite the paper as well.

## License

The source code is released under the MIT License. The recordings, audio clips, unpublished metadata, and other restricted research materials are not covered by this license. For reuse of the public result tables and figures, contact the authors and cite the paper.
