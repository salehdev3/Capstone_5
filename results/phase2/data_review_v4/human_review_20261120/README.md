# Uploaded named human mask review

The original uploaded ZIP is preserved byte-for-byte, with SHA256 in verification.json. All 71 geometry-queue images were reviewed by Yousef: 64 approved (51 train, 13 validation) and 7 rejected (5 train, 2 validation). The 64 approved label files and source/image identities match the frozen source and actual review records. Twenty-two retained raster-topology exceptions are explicitly recorded, not claimed as simple-polygon repairs.

The full V4 queue has 190 images. Exported requires_review=126 means 7 rejected plus 119 not reviewed in this pass; it does not mean the 71 decisions were lost. Rejected images are listed in retry_queue.csv. No new model score, training completion or complete dataset release is claimed. New validation references require versioned re-evaluation of both baseline and candidate, rather than comparing old-reference scores to new-reference scores as though identical.

Round 2 is a new seven-image draft version: SAM input 1024 and a positive interior point from the original reference clipped inside each human box, replacing the first attempt's box-center prompt. This is annotation assistance, not inference evaluation. New drafts still require human approval; old drafts and the 64 approvals are preserved. The prior reference may itself be imperfect, so the prompt does not guarantee a correct mask.
