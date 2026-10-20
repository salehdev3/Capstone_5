# العلاقة باللابات والتعديلات

هذه الدفاتر تكييف للابات المرفقة على بيانات Project 5؛ ليست نسخًا حرفية. أُبقيت الخطوات الأساسية مع إضافات تفرضها الوثيقة.

| المصدر | المستخدم في المشروع | التعديل وسببه |
| --- | --- | --- |
| Lab 1، فحص البيانات A | عرض RGB، الأبعاد، الهيستوغرام، قبل/بعد | صور CCT20 بدل Drone/Bird؛ نوضح أن الأصل المعروض نسخة تنزيل بحد أقصى 1024 |
| Lab 1، Preprocessor، الخلية 72 | `prepare` ثم augmentation ثم `finalize` | BGR→RGB، resize، ImageNet normalization، CHW؛ تعريف واحد مطابق في الأعضاء 1–3 |
| Lab 1، Split، الجزء C | `StratifiedGroupKFold` | المجموعة كاميرا كاملة؛ stratum هو مجموعة تسميات الصورة. تقسيم تقريبي 60/10/10/20، مع عرض الانحراف الفعلي |
| Lab 1، D | توازن الفئات، التدقيق، الضبابية وحد الخطأ | تدقيق train على مستوى الأحداث؛ استبعاد الأحداث ذات المشاكل المؤكدة؛ الضبابية للمراجعة فقط |
| Lab 1، Dataset/DataLoader، الخلية 104 | Dataset → DataLoader → batch preview | يدعم الصورة الكاملة أو قصاصة الصندوق البشري؛ augmentation على train فقط |
| Lab 1، Dataset Card | المصدر والترخيص والتنظيف والتقسيم والقيود | قيم من التشغيل وملاحظات يكتبها الفريق |
| Lab 2، ConvNN، الخلية 39 | ثلاث كتل Conv/BatchNorm/ReLU، pooling وطبقات fully connected | أبعاد Linear صريحة بدل LazyLinear وأدوات debug؛ dropout=.2؛ رأس بعدد فئات المهمة |
| Lab 2، حلقة التدريب | forward → loss → backward → optimizer.step | CrossEntropy وAdam؛ early stopping واختيار checkpoint على val_select |
| Lab 2، التقييم | منحنيات وconfusion matrix ودقة ومقاييس الفئات | top-k وECE وerror gallery وWilson وcamera bootstrap وفق الوثيقة |
| Lab 3، الخليتان 25 و34 | `microsoft/resnet-50` وتبديل classifier | frozen backbone ثم full fine-tuning، لنفس البيانات والمعالجة؛ تجميد حالة BatchNorm أيضًا في frozen |
| Lab 3، تتبع التجارب | W&B وcomparison | ست تجارب: مهمتان × scratch/frozen/full، مع تسجيل اختبار مستقل لكل تجربة |

الإضافات الخاصة بالمشروع: human-box species crops، عزل كاميرات الاختبار وbursts، تدقيق near-duplicates، فصل val_select عن val_calibrate، معايرة temperature وعتبة الحفاظ على الحيوانات، توثيق الأنواع غير المرئية في train، بصمات ملفات التسليم والمقارنة وإعادة تحميل checkpoints.

مصادر اللابات المحلية التي جرى الرجوع إليها:

- Day1_Lab_DataFoundations_DroneBird_solved.ipynb
- Lab2_Training_and_Evaluating_CNN_BigPicture.ipynb
- Day3_Lab_Transfer_Learning.ipynb

لا يُستخدم Lab 4 لتدريب detection في هذه المرحلة.
