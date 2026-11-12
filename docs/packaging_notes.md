# Packaging notes

The project title is taken from the supplied report. All five original authors remain credited.
The report is included unchanged as `docs/project_report.pdf`; its original filename
was `IEEE_Conference.pdf`. Its filename/layout alone is not evidence of conference publication.

## Organisation
- Renamed the course-code root and notebook to the project title.
- Renamed `part2_detection` to `detection`, `part3_recognition` to `recognition`,
  `end_to_end` to `pipeline`, and `outputs` to `data_preparation`; updated executable references.
- Kept final weights, the pretrained YOLOv8n starting checkpoint, all selected-data lists,
  metadata, per-image predictions, summary tables, training history and key visualisations.
- Removed the unused earlier `seven_head_residual_best.pt` checkpoint and its separate
  `training_history.csv`. The final spatial residual model and its history remain.
- Removed 20 redundant annotation-check PNGs. The notebook retains the code to regenerate them;
  detection galleries, failure examples and the full report remain available for review.
- Cleared cached notebook outputs and execution counts; removed automatic package installation
  and course-specific kernel metadata. Package installation is now an explicit setup step.
- Replaced machine-specific paths in detector training arguments and speed summaries with
  portable paths. Model files and measured numeric results are unchanged.
- An existing final detector can now be reused without requiring CUDA before the reuse check.
  Retraining still follows the original CUDA configuration.

## Scope of validation
The package is checked for archive integrity, Python/notebook syntax, saved split disjointness,
metadata counts, final-checkpoint presence and consistency of saved predictions and summaries.
No training or image-based inference was rerun: the original CCPD2019 images are not included.
Metrics and timings are existing group experiment records, not new measurements.
