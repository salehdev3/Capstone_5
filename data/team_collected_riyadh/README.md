# Team-Collected Riyadh Images

This folder contains 20 photographs supplied by the team for the local camera-trap classification extension. The original image files are retained unchanged.

`metadata_review.csv` contains proposed image-level labels and blank capture-time, site, session and reviewer fields. WhatsApp filenames are not treated as verified capture times. Fill those fields using the photographer's records.

`draft_boxes.json` contains proposed pixel-coordinate bounding boxes. These were prepared as annotation drafts and require human visual verification before use as ground truth. `review.html` displays the images with the proposed boxes. In particular, verify small birds, partially visible animals, all instances in multi-animal images and suspected repeat photos.

Preliminary labels: 13 cat images, one dog image, one bird image, and five images that appear empty. Keep the box label `cat` or `dog` only if it matches the supported species vocabulary; otherwise report it as an unknown class in species evaluation. No model evaluation has been run on these images.

These images are team-collected data. Keep this repository private and do not redistribute the photographs. Use them as a separate exploratory evaluation set unless the team creates a leakage-safe training split and reruns every comparison experiment.
