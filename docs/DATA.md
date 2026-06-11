# Data Availability

Full raw datasets and large generated outputs are not included in this GitHub repository because several files exceed normal GitHub repository size limits.

The complete local data package is stored externally on Google Drive.

Google Drive data link: [PASTE_GOOGLE_DRIVE_LINK_HERE]

## Restore Full Data Locally

After cloning this repository, download the full data package from Google Drive and place the downloaded folders/files back into the same project paths:

```text
data/raw/
data/external/
output/simulations/
output/brackets/
output/diagnostics/
```

These paths are intentionally ignored by Git so the full datasets can remain on your local machine without being committed to GitHub.

## Sample Data

The `data/sample/` folder contains small CSV samples with up to the first 100 rows from the local raw CSV files. These samples are tracked in Git for schema inspection, API review, and lightweight project understanding.

The sample files are not sufficient to rerun the full model pipeline. To reproduce the complete cleaning, rating, prediction, evaluation, and simulation workflow, restore the full data package from Google Drive first.
