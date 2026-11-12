# Evaluation definitions

## Accuracy

Detector-only evaluation (`detection/results/*_summary.csv`) chooses the highest-confidence
box per image, with confidence threshold 0.001. Localisation is correct only at IoU > 0.7.
Summing correct counts across the eight test subsets gives 9,052 / 10,000 = 90.52%.
Base alone is 2,976 / 3,000 = 99.20%; Base mAP@0.5 is 0.995.

Combined evaluation (`pipeline/results/end_to_end_summary.csv`) uses confidence threshold
0.25. Its detection rate is 99.64% and its localisation accuracy is 90.46%.
The threshold differs from detector-only evaluation, so the localisation totals differ slightly.

- Recognition-only accuracy: all seven characters match, using ground-truth boxes as crops.
- End-to-end exact-plate accuracy: all seven characters match after detection; 7,041 / 10,000.
- Strict end-to-end accuracy: both IoU > 0.7 AND all seven characters match; 6,635 / 10,000.
- Detection rate: an image has a box passing the threshold. This is not localisation accuracy.
- mAP is a precision/recall-based metric and is not the same as any of these image accuracies.

## Speed

The 42.51 FPS detector-only benchmark uses batch 1 on 500 Base images after 20 warmup images.
It includes repeated prediction calls and image handling. The combined notebook evaluation
uses detector chunks of 16 and records about 47.22 FPS across 10,000 images. Different batching
and execution paths mean the two numbers must not be compared as a detector-vs-recogniser
overhead measurement. Both are saved measurements on an RTX 3060 Laptop GPU, not guarantees
for other hardware or production workloads.

## Data boundaries

The 45,000-image protocol is a selected CCPD2019 experiment, not a full-dataset leaderboard
evaluation. Training, validation and formal test paths are disjoint. The development list is a
training subset. The report discusses a validation crop-margin experiment; its raw sweep logs
are not present in the supplied archive and have not been reconstructed or invented here.
