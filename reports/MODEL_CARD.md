# Camera-Trap Event Triage model card

The system assists human review of camera-trap events. It proposes animal locations, predicts species, outlines foreground and reports the maximum simultaneous count per event. The integrated release is phase2_results/integrated_species_cv/20261120T224149142114Z in the team Drive. The detector uses confidence 0.02 and NMS 0.30 followed by animal probability 0.25. A crop-guidance model provides a point to SAM at input size 224.

Historical evaluation included 554 retained frames and 508 known-class crops. Animal-frame recall was 0.9940, empty-frame removal 0.3019, species macro-F1 0.6513 and ECE 0.0382. Mean mask IoU was 0.662 on 40 historical frames. Event count MAE was 0.46875. Full-pipeline CPU p95 was 1.879 seconds over 20 calls on five fixed validation frames, after three warm-ups and excluding HTTP. Test observations informed development. Independent evaluation requires new camera events.

Small, obscured and edge-of-frame animals can be missed. Background can attract an incorrect box or mask. Species predictions can be wrong, particularly for rare species and new locations. Count represents animals visible together and does not identify unique individuals. Mask area is relative image area and does not establish physical size. Human review remains required for uncertain cases.
