# Team-collected data: mixed training release v2

The confirmed cities define the split: 14 Riyadh images supplement CCT20 training; 5 Tabuk images are reserved for test. All 5 Tabuk images form the local test before duplicate cleaning. Further cleaning or audit exclusions may reduce these counts. CCT20 supplies validation and calibration. City grouping avoids relying on inferred site independence.

All image labels and boxes were approved by Saleh. Collection dates, times, cities and review provenance are in metadata_review.csv. Pigeon and chicken map to the shared bird category; original labels remain recorded. Class support is computed from training data only. This small local sample is imbalanced: all Riyadh training images are animal-positive; Tabuk contains three empty images and two animal-positive images before cleaning. CCT20 retains both classes in training.

Start the notebook in the new CameraTrap_Phase1_Team_Mixed_v2 workspace. Member 1 finds this folder locally or obtains it from the public v1 branch (or uploads the folder to Colab). The existing notebook download cell uses the GITHUB_TOKEN secret in Colab; a public checkout or manual download does not require a token. The notebook imports local images before shared cleaning, leakage checks and label auditing. Retrain scratch, frozen and full models; do not reuse prior checkpoints or summaries. Outputs and dataset hashes are kept in the new workspace.

Evaluate the combined test and report the Tabuk subset separately. These two reports overlap and must not be counted as separate test samples. Never use test results to choose models, thresholds or the split. The earlier all-local-images external-test protocol is superseded.

The notebook records the collection protocol, source hashes, exclusions and retained class counts in the frozen release. Include these in the official report with the Dataset Card and actual experiment results. Bonus credit is assessed by the instructor, not guaranteed by a split ratio.

The 19 team photographs are intentionally published and publicly available in this folder on branch v1 of the public repository. review.html displays approved boxes; draft_boxes.json retains its historical filename. Confirmed site groups and uncertain assignments are documented in SITE_GROUPS.md.
