# Dataset Card — CCT20 / Phase 1
Source: https://lila.science/datasets/caltech-camera-traps plus team photographs from Riyadh and Tabuk.
Local protocol: Riyadh train; Tabuk test; validation/calibration from CCT20 only.
Local training is animal-only; the local test has very few frames and different class proportions.
Category mapping: pigeon/chicken -> bird; original species retained.
Retained team counts: {'train': 14, 'test': 5}.
Collection metadata and provenance: team_metadata.csv and team_collection.json.
Citation: Beery, Van Horn and Perona, Recognition in Terra Incognita, ECCV 2018.
Licence: CDLA-Permissive; follow course policy: do not redistribute images or commit them to Git.

## Tasks
Full-frame empty/animal classification; species classification on human-box crops.
No predicted boxes are used as ground truth. Positive frames without boxes remain in triage only.
Human/vehicle-labelled, conflicting or unlabelled frames are excluded by metadata.

## Acquisition and cleaning
Selected: 6041; retained after human audit: 6037.
Download failures: 0; exact duplicate IDs removed: 0.
Near-duplicate pairs across splits: 0.
Image maximum side: 1024; model input: 224 x 224; seed: 42.
Corrupt/failed-event and duplicate exclusions are recorded in cleaning_report.json.

## Split
StratifiedGroupKFold by camera, with joint image-label strata. Camera counts and class proportions below are measured, not assumed.
Whole bursts stay within one split; exact duplicate and event leakage checks pass.
Group constraints approximate class stratification; this is not the official CCT benchmark split.
               frames  cameras
split                         
test              639        4
train            3534       13
val_calibrate     527        2
val_select       1337        3

## Long-tailed class distribution (frame label occurrences)
          test  train  val_calibrate  val_select
label                                           
badger       1      4              0           0
bird         6    122             42           3
bobcat      58    322             73          27
cat         61    162             36         276
coyote     187    331             48          37
deer         3     12              0           0
dog         29    114             99          33
empty       53    319             13          91
fox          1      1              0           0
opossum    105    727            102         525
rabbit       7    364             63         177
raccoon     42    764             24          21
rodent       0     73              0           0
skunk       19     84             27           0
squirrel    67    137              6         147
Species crop counts: crop_counts.csv. Rare species may have little or zero support in held-out locations.
Training vocabulary is built from train crops only; unseen held-out species are reported by Member 2.

## Final stratification check
   label         split  count  frames   percent  overall_percent  difference_pp
  badger         train      4    3534  0.113186         0.082823       0.030364
  badger    val_select      0    1337  0.000000         0.082823      -0.082823
  badger val_calibrate      0     527  0.000000         0.082823      -0.082823
  badger          test      1     639  0.156495         0.082823       0.073672
    bird         train    122    3534  3.452179         2.865662       0.586517
    bird    val_select      3    1337  0.224383         2.865662      -2.641279
    bird val_calibrate     42     527  7.969639         2.865662       5.103978
    bird          test      6     639  0.938967         2.865662      -1.926695
  bobcat         train    322    3534  9.111488         7.950969       1.160519
  bobcat    val_select     27    1337  2.019447         7.950969      -5.931523
  bobcat val_calibrate     73     527 13.851992         7.950969       5.901023
  bobcat          test     58     639  9.076682         7.950969       1.125713
     cat         train    162    3534  4.584041         8.862018      -4.277977
     cat    val_select    276    1337 20.643231         8.862018      11.781214
     cat val_calibrate     36     527  6.831120         8.862018      -2.030898
     cat          test     61     639  9.546166         8.862018       0.684148
  coyote         train    331    3534  9.366157         9.988405      -0.622248
  coyote    val_select     37    1337  2.767390         9.988405      -7.221015
  coyote val_calibrate     48     527  9.108159         9.988405      -0.880245
  coyote          test    187     639 29.264476         9.988405      19.276071
    deer         train     12    3534  0.339559         0.248468       0.091091
    deer    val_select      0    1337  0.000000         0.248468      -0.248468
    deer val_calibrate      0     527  0.000000         0.248468      -0.248468
    deer          test      3     639  0.469484         0.248468       0.221016
     dog         train    114    3534  3.225806         4.555243      -1.329436
     dog    val_select     33    1337  2.468212         4.555243      -2.087030
     dog val_calibrate     99     527 18.785579         4.555243      14.230336
     dog          test     29     639  4.538341         4.555243      -0.016902
   empty         train    319    3534  9.026599         7.884711       1.141888
   empty    val_select     91    1337  6.806283         7.884711      -1.078428
   empty val_calibrate     13     527  2.466793         7.884711      -5.417918
   empty          test     53     639  8.294210         7.884711       0.409499
     fox         train      1    3534  0.028297         0.033129      -0.004832
     fox    val_select      0    1337  0.000000         0.033129      -0.033129
     fox val_calibrate      0     527  0.000000         0.033129      -0.033129
     fox          test      1     639  0.156495         0.033129       0.123365
 opossum         train    727    3534 20.571590        24.167633      -3.596043
 opossum    val_select    525    1337 39.267016        24.167633      15.099383
 opossum val_calibrate    102     527 19.354839        24.167633      -4.812794
 opossum          test    105     639 16.431925        24.167633      -7.735708
  rabbit         train    364    3534 10.299943        10.120921       0.179022
  rabbit    val_select    177    1337 13.238594        10.120921       3.117673
  rabbit val_calibrate     63     527 11.954459        10.120921       1.833538
  rabbit          test      7     639  1.095462        10.120921      -9.025459
 raccoon         train    764    3534 21.618563        14.096405       7.522157
 raccoon    val_select     21    1337  1.570681        14.096405     -12.525725
 raccoon val_calibrate     24     527  4.554080        14.096405      -9.542326
 raccoon          test     42     639  6.572770        14.096405      -7.523636
  rodent         train     73    3534  2.065648         1.209210       0.856438
  rodent    val_select      0    1337  0.000000         1.209210      -1.209210
  rodent val_calibrate      0     527  0.000000         1.209210      -1.209210
  rodent          test      0     639  0.000000         1.209210      -1.209210
   skunk         train     84    3534  2.376910         2.153387       0.223523
   skunk    val_select      0    1337  0.000000         2.153387      -2.153387
   skunk val_calibrate     27     527  5.123340         2.153387       2.969952
   skunk          test     19     639  2.973396         2.153387       0.820008
squirrel         train    137    3534  3.876627         5.913533      -2.036906
squirrel    val_select    147    1337 10.994764         5.913533       5.081231
squirrel val_calibrate      6     527  1.138520         5.913533      -4.775013
squirrel          test     67     639 10.485133         5.913533       4.571600
Rates count label presence per frame; multi-label class percentages need not sum to 100.
See class_balance_proportions.csv and species_crop_support.csv. Camera constraints can prevent exact balance.

## Label audit
Reviewed: 10 / 10; observed label errors: 1.
One-sided 95% upper error bound: 0.39416330243650466.
Sampling: one frame per randomly sampled train event. Same-camera dependence may make this bound optimistic.
Flagged training events are quarantined; original audit decisions remain recorded.

## Preprocessing
RGB, resize 224, ImageNet normalization. Random flip and brightness/contrast changes on train only.
All sections import image preprocessing and augmentation from preprocessing.py.

## Limitations and privacy
Main source is curated US wildlife data; local supplementation is too small to establish generalisation across Saudi locations. The empty fraction must be measured on this subset.
Grayscale is an infrared proxy, not verified nighttime. Blur is a review flag; no image is discarded solely for low Laplacian variance. No GPS publication; keep photos and review galleries private.
Test accuracy confidence intervals belong to Member 2 after training, not this untrained data notebook.

## Team observations
The dataset is strongly imbalanced across species and camera-based splits. Raccoon and opossum have many training examples, while fox, badger and deer have very few; some rare classes have no examples in one or more validation splits. The class distribution also changes between locations, which limits how well results from one split may generalize to another. Ten training events were reviewed in the label audit, and one label error was found; this small audit gives a wide 95% upper error bound of about 39.4%. The team-collected images add local examples from Riyadh to training and reserve Tabuk images for test, but this small collection is not representative of all Saudi camera-trap imagery.
