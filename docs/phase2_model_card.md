# Project 5 Phase 2 model card

**Status:** In development. No Phase 2 detector, segmenter, or integrated service has been evaluated yet.

**Intended use:** Batch review of camera-trap imagery by ecology teams. The proposed output is empty-frame filtering, animal boxes, species estimates, animal masks, and event-level counts with review images. Human review remains necessary for uncertain or high-impact records.

**Training data:** The Phase 1 CCT20 subset and the team-collected Riyadh training subset, using the existing camera/city-aware split. Human boxes only are eligible for detector ground truth. The small Tabuk subset remains held out. Segmentation masks require a separately documented human verification step.

**Evaluation protocol:** Select models on `val_select`, choose thresholds on `val_calibrate`, and run the locked test once. Report detection mAP50, mAP50–95, per-class AP, recall by object size, and operating-point precision/recall/F1. Report segmentation mask mAP50–95 or mean IoU and Dice, as appropriate to the trained model. Include event count error and served-system latency, throughput and memory.

**Known limitations:** Rare species, unseen locations, infrared imagery, small animals and the tiny local test subset may limit generalization. The Phase 1 frozen triage model missed its empty-removal target, and the species model missed its macro-F1 target. These findings remain relevant to the integrated system. Any image with identifiable people must be excluded or anonymized according to the dataset policy.

**Out of scope:** Surveillance of people, identification of individuals, estimating exact animal body size or distance without a calibrated camera, and autonomous decisions about conservation actions.

Model versions, operating thresholds, quantitative results and hardware benchmarks will be added only after completed runs.
