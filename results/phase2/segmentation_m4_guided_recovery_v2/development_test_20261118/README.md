# Recovery v2 development-test results

The exact uploaded files reproduce **mean IoU 73.99%** and **mean Dice 80.01%** over the same 40 test-image IDs. Policy SHA256 matches the released recovery v2 policy. Full precision and uploaded file hashes are preserved beside this report.

| Scope | v1 IoU | v2 IoU | v1 Dice | v2 Dice |
|---|---:|---:|---:|---:|
| Validation, 40 original references | 80.68% | 82.40% | 86.86% | 88.89% |
| Exposed development test, 40 original IDs | 58.02% | 73.99% | 62.02% | 80.01% |

Test mean IoU increases **15.97 percentage points**. Nine images improve, none worsen, and 31 retain their original IoU. The fallback runs in 10 images: nine recover some reference overlap and one still has zero overlap. Thirty images reach IoU >= 0.7. Two images still have no detector prompt; three have zero IoU overall. The large foreground false positive in `58678cd3` remains unchanged (foreground fraction 89.23%, IoU 1.89%).

The 70% mean-IoU target is met on this development test. Recovery v2 was designed after examining v1 errors on the same test images; the comparison is **not an independent estimate of generalization**. Do not report it as a fresh unseen test, omit failed images, or merge validation and test into one score.

No new training, mask editing or threshold search was performed for this documentation. The ZIP is the earlier frozen release; its embedded test status predates this evaluation. These files are the subsequent test evidence. Native standalone M5 AP does not describe the complete M4 + M5 + SAM pipeline; pipeline AP remains unmeasured.

Recommended retained release: `Segmentation_M4_Guided_Recovery_v2`. Preserve v1 outputs as the baseline and v2 as a separate version. Remaining limitations should appear in the failure gallery/model card. Any independent generalization claim requires a separate unseen evaluation set.
