# Visual review of 12 lower-IoU v2 cases

The assistant inspected the supplied originals and all 12 reference/prediction overlays. Proposed English cause/improvement fields are populated. Reviewer fields remain blank and the analysis is explicitly pending human confirmation. No reviewer identity or sign-off is invented.

Two cases exceed IoU 0.70 and are partial successes rather than outright failures. Some reference-only components appear over rocks or outside the annotated animal box; these are suspected reference-quality issues for independent adjudication, not permission to edit test labels or inflate existing results. The frozen masks, policies and scores remain unchanged.

Use the paired Colab confirmation helper after checking the images and comments. It asks for the actual reviewer name and explicit confirmation, checks current IDs/metrics/policy, backs up the old review CSV and updates review text only. Then rerun the evidence audit.
