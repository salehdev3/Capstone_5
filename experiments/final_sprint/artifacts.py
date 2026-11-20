"""Generate English submission drafts only from measured, version-bound evidence."""

from common import *
from runtime import DEMO_IDS
from pathlib import Path
import shutil, zipfile, platform, subprocess, html

TEAM = [
    ("Saud Abdullah Alfaiz", "225462", "Data and classification"),
    ("Saleh Mohammed Alanazi", "225486", "Detection"),
    ("Yousef Nafi Alanazi", "225457", "Segmentation"),
    ("Mohannad Mohammed Alqahtani", "225461", "Integration and report"),
]


def verdict(root):
    root, out, _, _ = context(root)
    test = out / "historical_test"
    d = read(test / "detection_metrics.json")
    s = read(test / "species_metrics.json")
    m = read(test / "segmentation/metrics.json")
    e = read(test / "event_metrics.json")
    b = read(out / "cpu_runtime/benchmark.json")
    g = read(out / "gpu_runtime/benchmark.json")
    t = PLAN["original_targets"]
    entries = [
        ("animal_recall", d["animal_recall"], t["animal_recall_min"], "min"),
        ("empty_removal", d["empty_removal"], t["empty_removal_min"], "min"),
        ("species_macro_f1", s["macro_f1"], t["species_macro_f1_min"], "min"),
        ("species_ece", s["ece"], t["species_ece_max"], "max"),
        ("mask_iou", m["mean_iou"], t["mask_iou_min"], "min"),
        ("event_count_mae", e["mae"], t["event_count_mae_max"], "max"),
        ("pipeline_cpu_p95_seconds", b["p95"], t["p95_cpu_seconds_max"], "max"),
    ]
    acceptance = [
        dict(
            metric=k,
            value=v,
            target=target,
            direction=direction,
            met=bool(v >= target if direction == "min" else v <= target),
        )
        for k, v, target, direction in entries
    ]
    csv_save(out / "acceptance.csv", acceptance)
    stretch = [
        dict(metric=k, value=v, target=0.8, met=bool(v > 0.8))
        for k, v in [
            ("species_accuracy", s["accuracy"]),
            ("species_macro_f1", s["macro_f1"]),
            ("mask_iou", m["mean_iou"]),
            ("box_precision", d["precision"]),
            ("box_recall", d["recall"]),
            ("box_f1", d["f1"]),
            ("empty_removal", d["empty_removal"]),
        ]
    ]
    save(
        out / "improvement_targets.json",
        dict(
            requested_rate_target="above .80; applies to rate metrics, not seconds/MAE/ECE",
            measurements=stretch,
            gpu_p95_seconds=g["p95"],
            gpu_latency_target=2.0,
            gpu_latency_met=g["p95"] <= 2.0,
            original_cpu_target_unchanged=True,
        ),
    )
    save(
        out / "decision.json",
        dict(
            performance_decision="GO" if all(r["met"] for r in acceptance) else "NO-GO",
            scope="Preset unattended deployment criteria; artifact completeness and course assessment are separate",
            acceptance=acceptance,
            policy_sha256=sha(out / "policy.json"),
            independent_test=False,
        ),
    )
    return acceptance


def failure_gallery(root, cache):
    from service import overlay
    from PIL import ImageDraw

    root, out, rows, classes = context(root)
    byid = {r["id"]: r for r in rows}
    dest = out / "failure_gallery"
    dest.mkdir(exist_ok=True)
    det = table(out / "historical_test/detection_per_image.csv")
    frames = {
        r["id"]: r for r in read(out / "historical_test/detection_predictions.json")
    }
    selection = sorted(det, key=lambda r: -(int(r["fn"]) + int(r["fp"])))[:12]
    records = []
    for r in selection:
        f = frames[r["id"]]
        cause = "Missed annotated animal" if int(r["fn"]) else "Unmatched predicted box"
        proposal = "Audit animal visibility, object scale and retained annotations; add reviewed training examples from unused training events."
        records.append(
            dict(
                task="detection",
                id=r["id"],
                observed_error=f"FP={r['fp']}; FN={r['fn']}",
                cause=cause,
                improvement=proposal,
                review_status="automated draft; no new human approval claimed",
            )
        )
        overlay(image_path(cache, r["id"]), f, dest / ("det_" + r["id"] + ".png"))
        target = dest / ("det_" + r["id"] + ".png")
        im = Image.open(target).convert("RGB")
        draw = ImageDraw.Draw(im)
        for b in byid[r["id"]]["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            draw.rectangle(
                (x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height),
                outline="cyan",
                width=2,
            )
        im.save(target)

    masks = sorted(
        table(out / "historical_test/segmentation/per_image.csv"),
        key=lambda r: float(r["iou"]),
    )[:12]
    frames = {
        r["id"]: r for r in read(out / "historical_test/segmentation/predictions.json")
    }
    for r in masks:
        records.append(
            dict(
                task="segmentation",
                id=r["id"],
                observed_error="Foreground IoU " + r["iou"],
                cause="Prediction/reference foreground mismatch; visual diagnosis required",
                improvement="Inspect missing body parts, background spill, prompt match and reference boundaries before approving corrections.",
                review_status="automated draft; no new human approval claimed",
            )
        )
        overlay(
            image_path(cache, r["id"]),
            frames[r["id"]],
            dest / ("mask_" + r["id"] + ".png"),
        )
        target = dest / ("mask_" + r["id"] + ".png")
        im = Image.open(target).convert("RGB")
        draw = ImageDraw.Draw(im)
        for line in (
            (root / "segmentation/labels/test" / (r["id"] + ".txt"))
            .read_text()
            .splitlines()
        ):
            polygon = np.array(list(map(float, line.split()[1:]))).reshape(-1, 2)
            points = [(float(x * im.width), float(y * im.height)) for x, y in polygon]
            if len(points) >= 3:
                draw.line(points + [points[0]], fill="cyan", width=2)
        im.save(target)
    csv_save(dest / "analysis.csv", records)
    # Include original human-reviewed analyses as historical evidence, never as approval of new errors.
    for source in [
        root / "phase2_results/detection_failure_analysis.csv",
        root
        / "phase2_results/member4_recovery_v2/segmentation_failure_analysis_v2.csv",
    ]:
        if source.exists():
            shutil.copyfile(source, dest / ("historical_" + source.name))


def document(path, title, sections, landscape=False):
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
        PageBreak,
        KeepTogether,
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4, landscape as wide

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    assets = Path(__file__).parent / "assets"
    if "ProjectSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ProjectSans", str(assets / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(
            TTFont("ProjectSansBold", str(assets / "DejaVuSans-Bold.ttf"))
        )
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "ProjectSans"
    styles["BodyText"].fontSize = 16 if landscape else 10

    styles.add(
        ParagraphStyle(
            name="SmallText",
            fontName="ProjectSans",
            fontSize=11 if landscape else 8,
            leading=15 if landscape else 11,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cover",
            fontName="ProjectSansBold",
            fontSize=28 if landscape else 22,
            leading=34 if landscape else 30,
            textColor=colors.HexColor("#123b57"),
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallHeader",
            fontName="ProjectSansBold",
            fontSize=11 if landscape else 8,
            leading=15 if landscape else 11,
            textColor=colors.white,
        )
    )
    styles["BodyText"].spaceAfter = 8
    styles["BodyText"].leading = 22 if landscape else 15
    pagesize = wide(A4) if landscape else A4
    doc = SimpleDocTemplate(
        str(path),
        pagesize=pagesize,
        rightMargin=36,
        leftMargin=36,
        topMargin=38,
        bottomMargin=35,
        title=title,
        author="Camera-Trap Project 5 Team",
    )
    story = []

    def para(text, style="BodyText"):
        return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])

    for index, section in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.append(para(section["title"], "Cover"))
        for text in section.get("text", []):
            story.append(para(text))
        for headers, data in section.get("tables", []):
            width = (pagesize[0] - 72) / len(headers)
            content = [[para(x, "SmallHeader") for x in headers]] + [
                [para(x, "SmallText") for x in row] for row in data
            ]
            table = Table(content, colWidths=[width] * len(headers), repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b57")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.lightgrey),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.whitesmoke, colors.white],
                        ),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
        for source, caption in section.get("images", []):
            from PIL import Image

            with Image.open(source) as im:
                w, h = im.size
            maxw = pagesize[0] - 72
            maxh = 240 if landscape else 290
            scale = min(maxw / w, maxh / h)
            story.extend(
                [
                    RLImage(str(source), w * scale, h * scale),
                    para(caption, "SmallText"),
                    Spacer(1, 10),
                ]
            )

    def footer(canvas, doc):
        canvas.setFont("ProjectSans", 8)
        canvas.setFillColor(colors.HexColor("#536878"))
        canvas.drawString(36, 20, "AI251P | Camera-Trap Event Triage | Project 5")
        canvas.drawRightString(pagesize[0] - 36, 20, str(doc.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build(root, cache):
    root, out, rows, classes = context(root)
    dest = out / "submission"
    dest.mkdir(exist_ok=True)
    acceptance = verdict(root)
    failure_gallery(root, cache)
    p = read(out / "policy.json")
    freeze = read(out / "freeze.json")
    test = out / "historical_test"
    d = read(test / "detection_metrics.json")
    s = read(test / "species_metrics.json")
    m = read(test / "segmentation/metrics.json")
    ap = read(test / "native_detector_ap.json")
    maskap = read(test / "segmentation/operating_mask_ap.json")
    b = read(out / "cpu_runtime/benchmark.json")
    gpu = read(out / "gpu_runtime/benchmark.json")
    e = read(test / "event_metrics.json")
    http = read(out / "http/demo_record.json")
    parity = read(out / "onnx/parity.json")
    selection = read(out / "mask_selection.json")["selected"]
    support = read(out / "data_support.json")
    history = table(out / "classifier/history.csv")
    ph1 = table(Path(__file__).parent / "assets/phase1_comparison.csv")
    decision = read(out / "decision.json")["performance_decision"]
    gallery = out / "failure_gallery"
    scopetext = "All final test results are historical development evidence on the original splits. Earlier project iterations exposed these test sets; no new independent generalization estimate is claimed."
    pct = lambda v: f"{100*float(v):.2f}%"
    rate_rows = [
        ["Native detector AP50", pct(ap["map50"]), "Low-floor detector only"],
        ["Native detector AP50:95", pct(ap["map50_95"]), "Low-floor detector only"],
        [
            "Filtered box precision / recall",
            pct(d["precision"]) + " / " + pct(d["recall"]),
            "Complete-box retained test frames",
        ],
        ["Filtered box F1", pct(d["f1"]), "IoU >=.50 matching"],
        ["Animal frame recall", pct(d["animal_recall"]), "Retained animal test frames"],
        ["Empty-frame removal", pct(d["empty_removal"]), "Retained empty test frames"],
        [
            "Species accuracy / all-14 macro F1",
            pct(s["accuracy"]) + " / " + pct(s["macro_f1"]),
            "Human-box known-class crops",
        ],
        ["Species ECE", f"{s['ece']:.4f}", "15 confidence bins"],
        [
            "Mask union IoU / Dice",
            pct(m["mean_iou"]) + " / " + pct(m["mean_dice"]),
            "All 40 original reviewed test masks",
        ],
        [
            "Operating mask AP50 / AP50:95",
            pct(maskap["mask_map50"]) + " / " + pct(maskap["mask_map50_95"]),
            "Accepted operating masks",
        ],
        [
            "Event count MAE",
            f"{e['mae']:.4f}",
            f"{e['events']} fully supervised events",
        ],
    ]
    criteria = [
        [
            a["metric"],
            f"{a['value']:.4f}",
            a["direction"] + " " + str(a["target"]),
            "Met" if a["met"] else "Not met",
        ]
        for a in acceptance
    ]
    benchrows = [
        [
            name,
            f"{v['p50']:.3f}",
            f"{v['p95']:.3f}",
            f"{v['p99']:.3f}",
            f"{v['throughput']:.3f}",
        ]
        for name, v in [("CPU", b), ("GPU", gpu)]
    ]
    examples = [out / "http" / (k + ".png") for k in DEMO_IDS]
    detfails = sorted(gallery.glob("det_*.png"))[:2]
    maskfails = sorted(gallery.glob("mask_*.png"))[:2]
    sections = [
        dict(
            title="Capstone Project — Phase 2",
            text=[
                "Automated Triage of Camera-Trap Imagery",
                "Project 5 | Four members | M4, M5 and M8",
                "Submission: 22 November 2026",
                "Repository: https://github.com/salehdev3/Capstone_5/tree/phase-2",
                "Experiment tracking: https://wandb.ai/saleh-alanazi-ministry-of-interior/camera-trap-phase2",
                f"Release {PLAN['version']} | policy SHA256 {freeze['policy_sha256']}",
                "The system detects animals, classifies species, estimates relative visible-body area, and aggregates event counts for wildlife review.",
            ],
            tables=[(["Team member", "Student ID", "Workstream"], TEAM)],
        ),
        dict(
            title="1. Team, data and acceptance",
            text=[
                "Project dataset: 6,037 images (6,018 CCT20 and 19 team-collected frames). Camera, event and decoded-pixel groups remain disjoint across train, validation-selection, validation-calibration and test. Animal frames with incomplete boxes are excluded from detector/count evaluation; difficult retained images are not removed.",
                "The 19 retained team-collected frames comprise 14 training and 5 test frames; they enter the same frozen workflow. The four-member workstream ownership is shown on the cover. Original Phase 1 submissions and weights/results are retained separately.",
                "A hundred detector samples and the original 230 reference masks were reviewed during the earlier workflow. V4 used 56 explicitly approved training-mask overrides. Original 40-image validation and 40-image test references remain unchanged.",
                scopetext,
            ],
            tables=[
                (
                    ["Split", "All frames", "Complete-box frames"],
                    [
                        [
                            split,
                            sum(r["split"] == split for r in rows),
                            sum(
                                r["split"] == split
                                and not (
                                    r["animal"]
                                    and (
                                        r.get("incomplete_boxes", True)
                                        or not r["boxes"]
                                    )
                                )
                                for r in rows
                            ),
                        ]
                        for split in ["train", "val_select", "val_calibrate", "test"]
                    ],
                ),
                (["Criterion", "Measured", "Fixed target", "Status"], criteria),
            ],
        ),
        dict(
            title="2. Phase 1 continuity and classifier improvement",
            text=[
                "The original Phase 1 report remains unchanged. Separately versioned Phase 2 recovery and optimization classifiers are compared with the new candidate on validation only. Historical comparisons do not imply identical weights.",
                "New candidate: ImageNet-pretrained ConvNeXt-Tiny, 14 animal species plus a background output. Only source-labelled empty training frames supply background crops. Two head-warmup epochs precede last-stage tuning; AdamW, cosine decay, augmentation, label smoothing, class weighting, dropout and early stopping limit fitting to the small training set.",
                "Species probabilities condition on the 14 animal outputs. Background filtering uses all 15 outputs. Temperatures are fitted only on val_calibrate. The selected species checkpoint is "
                + p["species_classifier"]["architecture"]
                + ".",
                "The fixed validation split contains 9 of the 14 species: the all-14 macro-F1 support ceiling is 9/14. All-class evaluation keeps the original denominator. Diagnostic accuracy gaps and class confusion document the observed generalization behaviour.",
            ],
            tables=[
                (
                    ["P1 regime", "Task", "Historical accuracy", "Historical macro F1"],
                    [
                        [
                            r["regime"],
                            r["task"],
                            pct(r["test_accuracy"]),
                            pct(r["test_macro_f1"]),
                        ]
                        for r in ph1
                    ],
                ),
                (
                    ["New candidate training", "Value"],
                    [
                        ["Completed epochs", len(history)],
                        [
                            "Last diagnostic train accuracy",
                            pct(history[-1]["train_subset_accuracy"]),
                        ],
                        [
                            "Last selection-validation accuracy",
                            pct(history[-1]["val_accuracy"]),
                        ],
                        ["Last accuracy gap", pct(history[-1]["accuracy_gap"])],
                    ],
                ),
            ],
        ),
        dict(
            title="3. M4 — Object detection",
            text=[
                "YOLO animal detection uses the already trained original and V3 detector candidates. Annotation conversion normalizes human CCT boxes into one animal class; input pixels and the frozen manifest are verified before use.",
                "The existing V3 YOLO11s experiment used 960-pixel input, AdamW, augmentation and native validation-based checkpoint selection. The original YOLO11n detector is retained as a candidate to avoid assuming that a larger network always improves triage.",
                f"Selected detector input {p['detector']['size']}; confidence {p['detector']['conf']}; NMS {PLAN['nms']}; maximum {PLAN['max_boxes']} boxes. Learned animal probability cutoff {p['detector']['animal_min']}. Confidence and background thresholds are selected jointly on all retained validation images. Test data do not select the new operating point.",
                "Native low-floor AP and filtered operating-point metrics are distinct; neither is presented as the other. Small-object, low-light and background errors are illustrated in the measured gallery.",
            ],
            tables=[(["Metric", "Measured", "Scope"], rate_rows[:6])],
        ),
        dict(
            title="4. M5 — Reference masks and segmentation",
            text=[
                "Original reference data contain 150 training, 40 validation and 40 test images. References originate from reviewed SAM proposals; shared teacher style is a limitation. V4 changes are training-only; all validation/test label hashes are checked. Self-touching polygons use the same foreground rasterization as earlier results.",
                "Candidates combine original/V3/V4 trained guides with direct learned masks, SAM-B or SAM2.1-Tiny. Prompts come from predicted boxes and predicted learned-mask interiors, with box-centre fallback. Reference boxes and polygons are never supplied as inference prompts.",
                f"Selected mask profile: {selection['name']}. The full pipeline runs the detector once, classifies detected crops in a batch, and reuses the detector result for refinement. Masks provide relative image foreground area, not physical body size, distance or body mass.",
                "Selection prefers validation IoU >=.80 with CPU p95 <=2 seconds, then the unchanged .70 mask target; if no configuration satisfies the joint constraint, the highest measured overlap is retained and the acceptance decision records the actual results.",
            ],
            tables=[
                (
                    ["Profile", "Validation IoU", "Validation Dice"],
                    [
                        [r["name"], pct(r["mean_iou"]), pct(r["mean_dice"])]
                        for r in read(out / "mask_selection.json")["candidates"]
                    ],
                ),
                (
                    ["Test measure", "Value"],
                    [
                        ["Union IoU", pct(m["mean_iou"])],
                        ["Dice", pct(m["mean_dice"])],
                        [
                            "Foreground-area MAE fraction",
                            f"{m['area_mae_fraction']:.5f}",
                        ],
                        ["Operating mask AP50", pct(maskap["mask_map50"])],
                        ["Operating mask AP50:95", pct(maskap["mask_map50_95"])],
                    ],
                ),
            ],
        ),
        dict(
            title="5. M8 — Integrated application and release",
            text=[
                "Contract: multipart POST /batch accepts image files plus one JSON metadata record per image (id, event and optional timestamp). Output includes animal boxes, species/confidence, review flags, masks/relative area, stage times and event summaries. Invalid metadata produces HTTP 400.",
                "Event count equals maximum simultaneous retained detections in an event. It is not unique-animal identity. The count evaluation covers only events with complete box supervision on every retained frame.",
                f"Real HTTP status 200; invalid request 400. Model-stage coverage: {http['coverage']}. The clean replay starts a fresh Python interpreter after installation and verifies hashes before inference.",
                "The installed release includes code, versioned policies, model hashes, training history, reference hashes, contracts and measured artifacts. A content-addressed local cache reduces Drive reads. Source and learned checkpoint snapshots support resumption.",
            ],
            tables=[
                (
                    ["Event measure", "Value"],
                    [
                        ["Count MAE", f"{e['mae']:.4f}"],
                        ["Evaluated events", e["events"]],
                        [
                            "Excluded incomplete-supervision events",
                            e["excluded_events"],
                        ],
                    ],
                )
            ],
        ),
        dict(
            title="6. ONNX export and parity",
            text=[
                f"The selected detector is exported as static FP32 ONNX, opset 17, input {p['detector']['size']}, without embedded NMS. Native and ONNX raw tensors, common NMS decisions and box metrics are compared on all {parity['frames']} retained validation images.",
                f"Raw tolerance: rtol .001 / atol .001. Matched-box IoU >= .999; coordinate error <= .1 pixels; confidence error <= .0001. Raw pass: {parity['raw_pass']}; decision pass: {parity['decision_pass']}; metric pass: {parity['metric_pass']}.",
                "Parity applies to the detector only. The learned animal/species classifier and SAM remain native PyTorch. The HTTP application uses the measured native pipeline; detector export is an independently verified deployment option.",
                "Forward-only native/ONNX CPU latency is recorded separately in onnx/benchmark.csv; it does not substitute for full-pipeline p95.",
            ],
        ),
        dict(
            title="7. Benchmark and deployment decision",
            text=[
                "The fixed five validation functional cases include previous misses and a negative frame. Benchmark: sequential batch 1, 3 warmups and 20 repeats, image decoding and all model stages included; HTTP transport excluded. GPU timing is synchronized. RSS is a cumulative process peak, not incremental model memory.",
                f"Preset unattended-deployment decision: {decision}. The CPU p95 target stays at 2 seconds; the additional GPU result does not replace that criterion. The full acceptance table is reported in Section 1.",
                "Results use fixed metric definitions and documented operating conditions. Research use retains human review for uncertain species or missing masks.",
            ],
            tables=[
                (["Backend", "p50 (s)", "p95 (s)", "p99 (s)", "Images/s"], benchrows)
            ],
        ),
        dict(
            title="8. Error analysis and demonstrations",
            text=[
                "The following examples are generated from actual historical-test predictions. New cause/improvement entries are automated review drafts; they do not claim a new named human review. Existing named human reviews are preserved as historical evidence.",
                "Detection: inspect missed small/occluded animals and unmatched boxes. Segmentation: inspect missing body parts, background spill, prompt mismatches and reference boundaries. Rare species require more reviewed, licensed examples from unused training events.",
            ],
            images=[
                (
                    f,
                    "Red: predicted box. Green: predicted mask. Cyan: original reference. Yellow: predicted species.",
                )
                for f in detfails[:1] + maskfails[:1]
            ],
        ),
        dict(
            title="9. Submission and self-assessment",
            text=[
                "M4 evidence: annotated conversion, existing fine-tuning runs, operating sweep, native AP, filtered metrics, error gallery and executed video inference.",
                "M5 evidence: reviewed source/override provenance, trained-guide comparison, frozen references, IoU/Dice/area error, accepted-mask AP and error gallery.",
                "M8 evidence: versioned integrated pipeline, ONNX parity, real HTTP request, fresh-process replay, CPU/GPU benchmarks, event counts, acceptance decision and model card.",
                "Delivery files: this report, Final_Presentation.pdf, Demo_Fallback.mp4, Detector_Video.mp4, code, policies, metrics and checkpoint snapshot. Generated report, slide deck, video and new failure-analysis drafts require team visual review before submission.",
                "Live presentation: 15–20 minutes, each member presents their own workstream. The recorded execution replay supplies a fallback. Open-world extension is not applicable to this four-member team.",
                "Limitations: historical exposed test sets, small class support, unseen camera shifts, teacher-assisted masks and relative-area proxies. The system supports wildlife review rather than unattended population estimates.",
            ],
            tables=[
                (
                    ["Workstream", "Evidence"],
                    [
                        ["M4", "Detection and threshold/AP evidence"],
                        ["M5", "Masks and overlap/AP evidence"],
                        ["M8", "Application, export, HTTP and timing evidence"],
                    ],
                )
            ],
        ),
    ]
    from charts import render

    figures = render(out, rows, classes)
    sections[2].setdefault("images", []).append(
        (
            figures["learning"],
            "Diagnostic training probe and selection-validation accuracy; different class support means the gap alone does not establish overfitting.",
        )
    )
    sections[1].setdefault("images", []).append(
        (
            figures["support"],
            "Original all-species support. Team-collected frames remain in their original splits.",
        )
    )
    if "confusion" in figures:
        sections[2]["images"].append(
            (
                figures["confusion"],
                "All 14 classes retained; zero-support rows are labelled, not excluded from macro-F1.",
            )
        )
    if "tradeoff" in figures:
        sections[3].setdefault("images", []).append(
            (
                figures["tradeoff"],
                "Operating thresholds selected on validation only. Dashed lines show the original triage targets.",
            )
        )
    size_table = test / "detection_size_recall.csv"
    if size_table.exists():
        sections[3]["tables"].append(
            (
                ["Size proxy", "Instances", "Matched", "Recall"],
                [
                    [
                        r["size"],
                        r["instances"],
                        r["matched"],
                        pct(r["recall"]) if r["recall"] else "n/a",
                    ]
                    for r in table(size_table)
                ],
            )
        )
    for name in ["BoxPR_curve.png", "PR_curve.png"]:
        path = test / "native_detector_ap" / name
        if path.exists():
            sections[3].setdefault("images", []).append(
                (
                    path,
                    "Native detector low-floor precision–recall curve; before learned background filtering.",
                )
            )
            break
    native_guide = test / "segmentation/native_mask_ap.json"
    if native_guide.exists():
        native = read(native_guide)
        sections[4]["tables"].append(
            (
                ["Native learned guide (before refinement)", "Value"],
                [
                    ["Mask AP50", pct(native["mask_map50"])],
                    ["Mask AP50:95", pct(native["mask_map50_95"])],
                ],
            )
        )
    document(dest / "Phase2_Report.pdf", "Project 5 — Phase 2", sections)
    save(dest / "report_content.json", sections)
    # A concise 15–20 minute deck following the guideline sequence.
    slides = [
        dict(
            title="Camera-Trap Event Triage",
            text=[
                "Project 5 — Phase 1 and Phase 2",
                "Wildlife review from images to event summaries",
                "15–20 minutes | Four contributors",
            ],
        ),
        dict(
            title="Problem, users and decisions",
            text=[
                "Users: conservation teams and wildlife reviewers.",
                "Keep animal events, reduce empty-image review, classify species and estimate visible foreground area.",
                "Acceptance criteria are fixed before evaluating the new release.",
            ],
        ),
        sections[1] | dict(title="Data and split strategy"),
        dict(
            title="Phase 1 results",
            text=[
                "Scratch versus transfer learning; original submitted results remain unchanged."
            ],
            tables=[
                (
                    ["Regime", "Task", "Accuracy", "Macro F1"],
                    [
                        [
                            r["regime"],
                            r["task"],
                            pct(r["test_accuracy"]),
                            pct(r["test_macro_f1"]),
                        ]
                        for r in ph1
                    ],
                )
            ],
        ),
        dict(
            title="Data improvements and classification",
            text=[
                "Source-labelled empty crops train the background output.",
                "Regularized transfer learning; validation-only checkpoint selection; separate calibration split.",
                f"Selected species model: {p['species_classifier']['architecture']}.",
                f"Historical crop accuracy {pct(s['accuracy'])}; all-14 macro F1 {pct(s['macro_f1'])}; ECE {s['ece']:.4f}.",
            ],
        ),
        dict(
            title="M4: detection and operating point",
            text=[
                f"Confidence {p['detector']['conf']}; NMS {PLAN['nms']}; animal-probability gate {p['detector']['animal_min']}.",
                f"Native AP50 {pct(ap['map50'])}; filtered box F1 {pct(d['f1'])}.",
                f"Animal recall {pct(d['animal_recall'])}; empty removal {pct(d['empty_removal'])}.",
            ],
        ),
        dict(
            title="M4: error example",
            images=(
                [
                    (
                        detfails[0],
                        "Actual historical-test prediction; inspect the paired error analysis.",
                    )
                ]
                if detfails
                else []
            ),
            text=[
                "Failure gallery includes every selected hard case; no difficult frame is removed from scoring."
            ],
        ),
        dict(
            title="M5: reference masks and results",
            text=[
                f"Selected profile {selection['name']}; validation IoU {pct(selection['mean_iou'])}.",
                f"Development-test IoU {pct(m['mean_iou'])}; Dice {pct(m['mean_dice'])}.",
                f"Accepted mask AP50 {pct(maskap['mask_map50'])}.",
                "Predicted boxes and learned points prompt refinement; relative area is an image proxy.",
            ],
        ),
        dict(
            title="M5: error example",
            images=(
                [
                    (
                        maskfails[0],
                        "Actual historical-test mask prediction; no GT prompts.",
                    )
                ]
                if maskfails
                else []
            ),
        ),
        dict(
            title="Integrated pipeline and HTTP demo",
            text=[
                "Image + metadata → detector → background/species classifier → guide/refiner → event table.",
                "Executed POST /batch, invalid-request check and fresh-interpreter replay.",
                "Show the live request; use Demo_Fallback.mp4 if needed.",
            ],
            images=[
                (examples[0], "Actual successful HTTP response rendered as an overlay.")
            ],
        ),
        dict(
            title="Deployment benchmark and ONNX",
            text=[
                f"ONNX raw/decision/metric parity: {parity['passed']} on {parity['frames']} validation images.",
                f"CPU p95 {b['p95']:.3f} s; GPU p95 {gpu['p95']:.3f} s.",
                f"Event count MAE {e['mae']:.3f} on {e['events']} complete-box events.",
                "Detector-forward timings and full-pipeline timings have separate scopes.",
            ],
        ),
        dict(
            title="Acceptance and deployment decision",
            text=[f"Preset decision: {decision}", scopetext],
            tables=[(["Criterion", "Value", "Target", "Status"], criteria)],
        ),
        dict(
            title="Contributions and next improvements",
            tables=[(["Member", "Contribution"], [[r[0], r[2]] for r in TEAM])],
            text=[
                "Priorities: rare-species coverage, new camera evaluation and latency/quality tradeoffs.",
                "Original Phase 1 artifacts remain separately versioned.",
            ],
        ),
        dict(
            title="Live demonstration",
            text=[
                "Load the release from Drive using the one-cell notebook.",
                "Send five fixed validation images to the local HTTP service.",
                "Inspect event counts, species, masks and review flags.",
                "Fallback: recorded execution replay from the same real HTTP response.",
            ],
        ),
    ]
    # Keep slides concise; replace the full report data section with a compact version.
    slides[2] = dict(
        title="Data and splitting",
        text=[
            "6,018 CCT20 + 19 team-collected frames; camera/event/pixel groups separated.",
            "Separate train, selection-validation, calibration and historical test.",
            "Detection uses complete-box frames; masks use 150 / 40 / 40 images.",
            "Original validation/test references are preserved; test results are development evidence.",
        ],
    )
    document(
        dest / "Final_Presentation.pdf", "Project 5 — Final Presentation", slides, True
    )
    save(dest / "slide_content.json", slides)
    notes = [
        "1–2. Problem and users: 2 minutes.",
        "3. Data (Saud): 2–3 minutes.",
        "4–5. Phase 1 and classifier improvements (Saud): 2–3 minutes.",
        "6–7. Detection (Saleh): 2–3 minutes.",
        "8–9. Segmentation (Yousef): 2–3 minutes.",
        "10 and 14. HTTP demonstration (Mohannad): 2–3 minutes.",
        "11–13. Benchmark, decision and contributions (Mohannad): 2–3 minutes.",
    ]
    (dest / "PRESENTATION_NOTES.txt").write_text("\n".join(notes))
    (dest / "MODEL_CARD.md").write_text(
        "# Camera-Trap Event Triage\n\nIntended use: human-assisted wildlife image/event review.\n\nFrozen policy: "
        + freeze["policy_sha256"]
        + "\n\n"
        + scopetext
        + "\n\nSpecies metrics use known-class human-box crops, not end-to-end species accuracy. Counts are maximum simultaneous retained detections; masks estimate relative image area. Review remains required for uncertain classifications and missing masks.\n\nThe CPU criterion is unchanged; GPU results are separately labelled. No identity tracking or physical-size estimate is claimed. See acceptance.csv, policy.json and the version-bound measured reports.\n"
    )
    card = [
        "# Project 5 dataset card",
        "",
        f"Frozen manifest SHA256: {MANIFEST_SHA}",
        "",
        "Sources: 6,018 CCT20 frames and 19 team-collected frames. Original source terms apply; original images remain in the project Drive.",
        "Camera, event and decoded-pixel groups are separated by the frozen metadata. The 19 retained local frames stay in their original 14-train/5-test split.",
        "Selection uses val_select; temperatures use val_calibrate. Historical test sets were exposed during prior iterations and are development evidence, not a new independent estimate.",
        "Positive frames with incomplete boxes are excluded only from complete-box detection/count evaluation. All original 40-image validation and test mask references are preserved.",
        "",
        "| Species | Train crops | Selection-validation crops | Calibration crops | Historical test crops |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in classes:
        counts = [
            sum(
                b["species"] == name
                for r in rows
                if r["split"] == split
                for b in r["boxes"]
            )
            for split in ["train", "val_select", "val_calibrate", "test"]
        ]
        card.append("| " + name + " | " + " | ".join(map(str, counts)) + " |")
    (dest / "DATASET_CARD.md").write_text("\n".join(card) + "\n")
    print("Generated report, slides and model card:", dest, flush=True)
