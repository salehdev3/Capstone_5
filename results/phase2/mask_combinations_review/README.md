# Executed nine-configuration mask comparison

Original/new detector x V3/V4 guide x SAM-B/SAM2.1-t, plus the released baseline, were compared on the same original 40 validation references. Every mean IoU/Dice/time value is reconstructed from per-image CSVs; weight identities match each policy and fixed plan.

Best IoU .8246557321 / Dice .8894653232 uses old detector + V3 guide + SAM-B. Baseline IoU .8239635728 / Dice .8889444578. Difference .06921593 percentage points: 15 images improve, 19 worsen, 6 unchanged. This small validation difference does not establish a meaningful independent gain; >90% IoU was not attained.

Retain the deployed release and prioritize classification data coverage. Further blind mask training is not justified by these results. GPU mask-helper means are not full CPU p95. Uploaded source JSON/CSV/logs are preserved here; REVIEW_SUMMARY.json records provenance, checks and recommendation. No new independent test or deployment is claimed.
