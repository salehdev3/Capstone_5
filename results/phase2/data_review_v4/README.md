# Data coverage and reference review audit

Actual frozen-source audit: 6,037 frames, 14 species; crop, event and camera counts are in species_coverage.csv and species_by_camera.csv. Validation lacks badger, deer, fox, rodent and skunk. Training contains 1 fox crop, 4 badger crops and 11 deer crops. Planning floor 100 is a collection aid, not a guarantee.

The review queue covers 150 training and 40 validation images; 71 images contain 73 geometric findings. Twelve geometry-only suggestions preserve original raster overlap >= .996046. Suggestions are unapproved. Visual semantics must be checked on original images; intentional topology can be retained only with an explicit named exception and exact foreground preservation. No test masks are included. All old references remain untouched.

Run notebooks/Phase2_Data_Review_V4.ipynb on CPU: initialize, then review one image per cell execution. Each record is saved to Drive. Export partial/final records for inspection. Public catalogue discovery is supplied for Colab because storage endpoints returned HTTP 403 here. No new images, named reviews or improved model score are claimed.
