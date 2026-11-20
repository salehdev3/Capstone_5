# Actual V4 training and validation review

56 named approved training-mask replacements were applied; all 40 validation label SHA256 hashes match the original release. Training stopped at epoch 21; native mask AP selected epoch 13 (AP50 .83604, AP50–95 .56458 in the logged validator). The selected checkpoint is bb89f9a6ec1b19a48c60de266f0cf08171ca5798bdd34e75996f1b0eed3eaaa0.

| Original-reference validation comparison | Mean IoU | Dice |
| --- | ---: | ---: |
| V3 standalone best threshold | .655042 | .720052 |
| V4 standalone best threshold | .624500 | .710004 |
| Reproduced original full pipeline | .823964 | .888944 |
| Best tested SAM2.1-t pipeline (1024, quality floor 0) | .797450 | .872146 |

The tested SAM2 pipelines use the V3 raw winner as their learned guide, because the script first selected by standalone IoU. They therefore do not establish V4-guided pipeline quality. The original per-candidate policy dictionaries inherited old weight filenames/hashes; they are source-policy context, not valid new deployment manifests. REVIEW_SUMMARY.json records the effective guide and this limit; original uploaded outputs are preserved without relabelling.

The follow-up compare_combinations.py fixes model identities in each configuration and evaluates both old/new detector and V3/V4 guide with SAM-B/SAM2.1-t. It uses nine fixed configurations and the same original 40 references. No training, GT-box prompts, test rerun or new deployment is part of that comparison. It remains validation development tuning, with independent assessment still needed for generalization claims.

GPU mask-helper mean times are not full CPU species/HTTP p95. Curated source JSON/CSV/YAML are archived here; training-image mosaics and binary weights are not duplicated. No >90% mask result was achieved in this uploaded experiment.
