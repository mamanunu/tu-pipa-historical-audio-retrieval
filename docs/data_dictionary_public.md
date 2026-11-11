# Public results: field guide

This repository provides aggregate results without source audio or performer-level details, along with paper figures that do not include identity mappings.

- `analysis_summary.txt`: text summary of the paper's main statistics.
- `pca_summary.csv`: explained and cumulative variance ratios for the principal components of each representation.
- `retrieval_metrics.csv`: aggregate retrieval metrics and intervals by representation, task, and query category.
- `retrieval_representation_differences.csv`: paired P@5 differences and intervals between the three representations.
- `retrieval_top5_stability.csv`: overlap among the Top-5 candidate sets returned by different representations.
- `aggregation_sensitivity_summary.csv`: macro-averaged results for minimum segment distance, mean segment distance, and recording-prototype aggregation.
- `sensitivity_low_energy.csv`: changes in aggregate results after excluding low-energy segments.
- `sensitivity_overlap.csv`: changes in aggregate results after excluding recordings with overlapping windows.
- `figure_pca_abc.png`: two-dimensional PCA spaces for the three representations; genre labels are shown, but segment, title, and performer identities are not.
- `figure_corpus_workflow.png`: overview of the research workflow.

The aggregate results cannot be used to reconstruct the source audio and cannot replace recomputation of features, rankings, or bootstrap confidence intervals from the full audio inputs.
