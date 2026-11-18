# M4-guided segmentation release

The ZIP contains the exact verified M4 detector, continued-fine-tuned M5 segmenter, SAM refinement inference code, policies and validation evidence. SAM ViT-B downloads on first inference and is checked by SHA256.

Validation on all 40 original reviewed images: mean foreground-union IoU **80.68%**, mean Dice **86.86%**. References and splits were preserved. These are repeatedly selected development validation results; independent test performance has not been measured. They must not be presented as standalone YOLO or test results.

Open [the Drive installer in Colab](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/phase-2/notebooks/Save_Segmentation_M4_Guided_To_Drive.ipynb) and run its cell. It downloads this ZIP, checks the archive and all release files, and installs in `MyDrive/CameraTrap_Phase2/releases/Segmentation_M4_Guided`. No GPU is required for installation.

The installed release includes `Import_M4_Guided_Segmentation.ipynb`, `predict_refined.py`, `refinement_policy.json`, `MODEL_CARD.md`, weights and metrics. For inference, install `ultralytics==8.3.221` and run the helper with `--images` and `--output`. Read the model card for test scope, training provenance and SAM-assisted reference limitations.

All original checkpoints and earlier test results are preserved. The package does not contain raw camera-trap images or test predictions.
