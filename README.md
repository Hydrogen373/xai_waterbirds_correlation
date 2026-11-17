# Waterbirds VGG11 shortcut experiment

Train VGG11 from random initialization for 50 epochs and track classification,
linear readability of bird/background labels, and Grad-CAM over time.

## Run

Install the packages in `requirements.txt`, then run:

```powershell
python experiment.py --output-dir runs/vgg11_scratch
```

The default is 50 epochs, batch size 32, and seed 42. A CUDA GPU is selected
automatically when available. The complete VGG11 is large; reduce `--batch-size`
if GPU memory is insufficient. Weight snapshots every five epochs use roughly
6 GB of disk in total. `--epochs 1` can be used for a smoke run.

The script downloads `grodino/waterbirds` through Hugging Face Datasets. It
stratifies the official train split by `(label, place)` into 80% training and
20% validation. The official test split is read only after training. Input
images are resized to 256 pixels on the short side, center-cropped to 224,
normalized from `[0, 1]` to `[-1, 1]`, and horizontally flipped only during
training. No pretrained weights or ImageNet normalization statistics are used.

## Outputs

`epoch_metrics.csv` contains per-epoch overall and four-group accuracies,
worst-group accuracy, aligned/conflicting accuracy and their difference, plus
linear-probe accuracy for `label` and `place`. Probe fitting uses the model's
last hidden activation, with equal training weight for each of the four groups.
The fixed ten-image Grad-CAM sample is saved at epoch 0 and every five epochs.
`epoch_curves.png`, `gradcam_timeline.png`, checkpoints, split indices, settings,
and final `test_metrics.json` are written to the output directory. The output
directory must be empty before starting a run. `gradcam_changes.csv` records
the mean absolute pixel difference between each normalized heatmap and its
previous five-epoch heatmap; it measures spatial change, not attribution strength.

High `place` probe accuracy means the representation contains readable
background information; it does not prove that the classifier uses that
information for its decisions. Read it alongside conflicting-group accuracy
and the qualitative Grad-CAM panels. The train-derived validation split retains
the dataset's strong correlation between bird type and background, so minority
group sample counts are included in every evaluation row.
