# Data Availability

Full raw datasets and large generated outputs are not included in this GitHub repository because several files exceed normal GitHub repository size limits.

The complete local data package is stored externally on Google Drive:

```text
https://drive.google.com/file/d/1bJnvtBXX1PASxo_KnYHa7Js0-cYG1v1I/view?usp=drive_link
```

## Restore Full Data Locally

After cloning this repository, download the full data package from Google Drive and place the downloaded folders/files back into the same project paths:

```text
data/raw/
data/external/
output/simulations/
output/brackets/
output/diagnostics/
```

If a `data/sample/` folder is included in the downloaded package, place it under:

```text
data/sample/
```

These paths are intentionally ignored by Git so the full datasets can remain on your local machine without being committed to GitHub.

## Reproducibility Note

The GitHub repository contains the source code, configuration, documentation, API code, and small demo outputs. To reproduce the full cleaning, rating, prediction, evaluation, and simulation workflow, restore the full data package from Google Drive first.
