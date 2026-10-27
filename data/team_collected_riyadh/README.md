# Team-Collected Riyadh Images

This folder contains 20 photographs supplied by the team for the local camera-trap classification extension. The original image files are retained unchanged.

`metadata_review.csv` contains proposed image-level labels and blank capture-time, site, session and reviewer fields. WhatsApp filenames are not treated as verified capture times. Fill those fields using the photographer's records.

`draft_boxes.json` contains proposed pixel-coordinate bounding boxes. These were prepared as annotation drafts and require human visual verification before use as ground truth. `review.html` displays the images with the proposed boxes. In particular, verify small birds, partially visible animals, all instances in multi-animal images and suspected repeat photos.

Preliminary labels: 13 cat images, one dog image, one pigeon image, one chicken image, and four images that appear empty. Keep the box label `cat` or `dog` only if it matches the supported species vocabulary; otherwise report it as an unknown class in species evaluation. No model evaluation has been run on these images.

These images are team-collected data. Keep this repository private and do not redistribute the photographs. Use them as a separate exploratory evaluation set unless the team creates a leakage-safe training split and reruns every comparison experiment.

## Notebook evaluation

Member 4 includes `Team-Collected Riyadh Data — External Evaluation`. Set `LOCAL_DATA_ROOT` to this folder in the Colab runtime. A standalone notebook cannot download a private dataset without your authenticated repository checkout or upload.

Review all images and all boxes, correct mistakes and missing instances, and only then set the corresponding metadata and annotation `status` fields to `verified`. Fill the reviewer, site and session fields; record actual capture dates and times where known. Do not use WhatsApp transfer times as capture dates. Images 17 and 20 may belong to the same session. Review the people visible in image 1 before sharing.

This set remains entirely in `local_external_test`; no local images influence training, model selection, calibration or threshold selection. Duplicate screening quarantines exact/near duplicate local images and potential overlap with CCT20. Results and a Dataset Card are saved under `submission/local_riyadh/` and copied into the evidence archive. The official report should describe collection, annotation review, cleaning, class balance, the separate local results and the limitations of this small sample. No bonus score is guaranteed.

## Review approval

On 2026-10-27, Saleh confirmed all image labels, box boundaries and animal-instance coverage. Metadata and annotation statuses are now verified. The historical filename `draft_boxes.json` is retained for notebook compatibility; its records are human-verified. Earlier pending-review instructions describe the preparation workflow. Site-group uncertainty remains documented separately. Model evaluation has not yet been run on these images.

## City correction

Images 1, 3, 4, 5 and 11 are from Tabuk, as confirmed by the photographer. The other 15 remain recorded in Riyadh. Folder and RYD identifiers are retained for compatibility. The prior shared-site inference for images 11 and 12 was withdrawn.
