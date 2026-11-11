# Method parameters

## Inputs and segmentation

- Core corpus: 54 mono WAV recordings.
- Genre labels: 45 Hua Diao recordings and 9 Bei Gong Diao recordings.
- Five four-second segments are selected from each recording, for 270 segments in total.
- Fixed relative positions: 0.20, 0.35, 0.50, 0.65, and 0.80 within the central 2%–98% interval of each recording; four-second windows are centered on those positions.
- Every segment included in the paper's analysis was reviewed by listening.

## Audio processing

- Analysis sample rate: 22,050 Hz.
- Analysis channel: mono.
- Frame length: 2,048 samples.
- Hop length: 512 samples.
- Analysis window: Hann.
- Mel filter bands: 40.
- Mel-filter frequency range: 50–8,000 Hz.

## Three feature representations

- A: temporal mean and population standard deviation for 13 MFCCs, for 26 dimensions.
- B: representation A plus the mean and population standard deviation of six spectral-shape descriptors, for 38 dimensions.
- C: representation B plus the mean and population standard deviation of log-RMS, for 40 dimensions.

The spectral-shape descriptors are centroid, spread, skewness, kurtosis, 85% roll-off, and flatness in dB. Log-RMS is calculated as 20 log10 of the RMS for each Hann-windowed frame.

## Scaling and distance

- Each feature dimension is robust-scaled using its median and interquartile range.
- An interquartile range below `1e-12` is replaced with 1.0.
- PCA uses centered NumPy SVD without whitening.
- Nearest-neighbor distance is Euclidean distance in the scaled feature space.
- Random seed: 20261004.

## Retrieval and evaluation

- Query: one four-second segment.
- Main candidate representation: the arithmetic mean of the five segment vectors for each candidate recording (the recording prototype).
- The query's source recording is excluded, leaving 53 candidate recordings.
- Main metrics: P@5, category-specific results, and a macro-average that weights the two genres equally.
- Sensitivity checks: paired representation differences, Top-5 set overlap, candidate aggregation method, low-energy segments, and overlapping windows.
