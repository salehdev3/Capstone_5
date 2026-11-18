# Recovery v2

Validation on the same 40 frozen references: IoU **82.40%**, Dice **88.89%**. Original pipeline: IoU 80.68%, Dice 86.86%. The fallback is used in two validation images and leaves all primary accepted outputs unchanged.

This is a new development version, motivated by v1 test failure analysis. It has no new test score yet. It does not fix the large foreground false positive or the two diagnostic images without detector prompts. No weights were retrained and no references were edited.

The package includes the same M4 and M5 weights as v1, an explicit fallback policy, inference helper and validation comparison. SAM downloads on first inference. Low-confidence masks require review.

[Open installation and development-test notebook](https://colab.research.google.com/github/salehdev3/Capstone_5/blob/phase-2/notebooks/Evaluate_M4_Guided_Recovery_v2.ipynb). Original v1 test results are not overwritten.

## Subsequent development-test evidence

The user evaluated this exact policy on the original exposed 40-image test set: **mean IoU 73.99%, Dice 80.01%**, compared with v1 58.02% / 62.02%. Nine images improve, none worsen. This is development testing after v1 error analysis, not a new independent test.

[Uploaded evidence, per-image comparison and verification](../../results/phase2/segmentation_m4_guided_recovery_v2/development_test_20261118/README.md). The ZIP remains unchanged and preserves its pre-evaluation record; the linked results document the later evaluation.
