# Waterbirds correlation experiment

Train three VGG11 classifiers from random initialization for 50 epochs each.
The training and validation images have bird/background correlations of
`0.95`, `0.80`, and `0.65`; every model is evaluated against the same official
Waterbirds test split, whose correlation is approximately `0.5`.

## Data and run

The experiment needs the original CUB-200-2011 images, segmentation masks,
`images.txt`, and `train_test_split.txt`, plus Places365 `data_256` images in
`b/bamboo_forest`, `f/forest/broadleaf`, `o/ocean`, and `l/lake/natural`.
The defaults point to the locally available sibling project. Override them
when running elsewhere:

```powershell
python -m pip install -r requirements.txt
python experiment.py --cub-root C:\path\to\CUB_200_2011 --places-root C:\path\to\places365 --output-dir runs/vgg11_correlations
```

`--epochs` defaults to 50, `--batch-size` to 32, and `--seed` to 42. CUDA is
selected automatically when available. Use `--resume` with the same settings
and output directory after an interrupted run. The complete VGG11 is large;
reduce `--batch-size` if GPU memory is insufficient. The script downloads the
official `grodino/waterbirds` test split through Hugging Face Datasets.

The 5,994 bird images in the original CUB training partition are split once,
stratified by bird label, into 4,795 train and 1,199 validation bird images.
All three conditions use these same bird identities and the same number of
examples. A single fixed land background and water background are assigned
to each bird; the correlation condition selects which is composited. The
background image pools for train and validation are disjoint. Each condition
sets the aligned fraction separately within landbirds and waterbirds, rounded
to a whole number of images. This preserves the original class proportions.
The official test birds come from the separate original CUB test partition.

## Metrics and outputs

Each condition directory (`corr095`, `corr080`, `corr065`) contains one
`epoch_metrics.csv` row per epoch, a `final.pt` model, a resumable `latest.pt`,
and three GradCAM thumbnails per epoch. `validation_accuracy` is ordinary
accuracy on the condition-matched validation split. `test_accuracy` is
ordinary accuracy on the common official test split. Worst-group accuracies
are also recorded.

The bird and background linear probes use the classifier's last hidden
activation. They are fitted on the condition's training images using equal
weight for each of the four `(bird, background)` groups. Their reported
accuracy is the average of the four group accuracies on the official test
images. A high background probe score indicates that the representation
contains background information; it does not alone establish that the
classifier relies on it.

At the top level, `epoch_metrics.csv` combines all 150 metric rows and
`epoch_curves.png` shows four metrics for all three conditions.
`gradcam_grid.png` has epochs as 50 rows and correlation/sample
combinations as nine columns. Samples a, b, and c are fixed official test
images shared by all three models: an aligned waterbird, a waterbird on land,
and a landbird on water. GradCAM always targets the true bird class.
`gradcam_samples.json` stores their test indices. `sample_a.png`,
`sample_b.png`, and `sample_c.png` show the original model inputs.
`data/` contains generated composites, per-condition manifests, and
`group_counts.json` with the realized
correlation for each bird label.

Run the offline checks with:

```powershell
python -m unittest discover -s tests -v
```
