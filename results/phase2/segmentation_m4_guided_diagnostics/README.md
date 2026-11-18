# Supplied test diagnostics

The two supplied files contain 40 matching IDs and the original frozen pipeline policy hash. The arithmetic means reproduce the reported test scores.

12 images have no accepted masks and IoU zero: 2 have no M4 prompt at the 0.045 floor; 10 have prompts but no accepted output. The supplied JSON retains accepted masks only, so the latter cannot yet be attributed specifically to confidence filtering, SAM quality or absence of SAM output.

Image `58678cd3-23d2-11e8-a6a3-ec086b02610b` has IoU 0.01886 and predicted foreground 0.89228. A large detected box covers 83.87% of the image; its SAM output has quality score 1.0104. SAM quality is a predicted score and is not measured overlap or a calibrated probability. This single image contributes 87.47% of summed absolute foreground-area errors. Visual review is needed for object identity and reference adequacy.

24/40 images reach IoU >= 0.7. All 40 remain in the reported 0.58022 aggregate. No sample exclusions, reference edits or threshold changes were made.

Next: replay the existing fixed inference policy with rejected-prompt logging. Diagnose before proposing changes; future adaptations based on these test errors are development work, requiring a separate unseen test for independent claims.
