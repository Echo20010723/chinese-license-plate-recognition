# File naming and migration

Python modules and regular documents use descriptive `snake_case` filenames.
Standard community entry files (`README.md`, `CONTRIBUTING.md`, `THIRD_PARTY_NOTICES.md`)
retain their conventional names. Notebook work is under `notebooks/`; reports and
guides are under `docs/`.

| Previous path | Current path |
| --- | --- |
| `parse_tool.py` | `ccpd_annotations.py` |
| `review_results.py` | `validate_results.py` |
| `Car_License_Plates_Detection_and_Recognition.ipynb` | `notebooks/ccpd_workflow.ipynb` |
| `PROJECT_OVERVIEW.html` | `docs/project_overview.html` |
| `report/Project_Report.pdf` | `docs/project_report.pdf` |
| `docs/EVALUATION.md` | `docs/evaluation.md` |
| `docs/PACKAGING_NOTES.md` | `docs/packaging_notes.md` |
| `docs/VALIDATION.md` | `docs/validation.md` |
| `recognition/checkpoints/seven_head_spatial_residual_best.pt` | `recognition/checkpoints/seven_head_ccpd2019.pt` |

Executable imports, notebook code, CLI defaults, HTML links and documentation were
updated together. The original PDF content and model tensor values were not edited.

Experiment result CSVs and detector run folders retain their descriptive run/seed names
so that recorded metrics remain traceable. The detector's `weights/best.pt` name
follows the standard Ultralytics training output layout.
