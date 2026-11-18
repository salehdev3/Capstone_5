# First user HTTP demo

HTTP status 200 confirms transport and handler execution. All three frame outputs have zero detections, zero accepted masks and no events. Classification was therefore not exercised, and SAM returned no masks. This is not successful full-stage demonstration evidence.

The frozen manifest labels the first two frames as animal (coyote and squirrel), but both lack complete box annotations. They were outside the reviewed complete-box detection subset. The third is an empty frame. Outputs show misses relative to frame-level labels; visual/source review is still needed. Do not infer positive-path latency from their approximately 0.4-second empty-path timings.

The revised demo retains all three cases and adds two previously detected, annotated validation cases (cat and opossum) to exercise the positive branches. The selection is explicit functional coverage, not a representative or independent accuracy sample. No policy or model weights were changed. Original evidence is preserved here.
