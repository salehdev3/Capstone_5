"""Bounded, resumable one-command orchestration; never convert missing evidence into a pass."""

from common import *
import argparse, sys, subprocess, time, traceback, os, shutil, zipfile


def prepare(root, cache):
    from train import prepare as images
    from runtime import sources

    root, out, rows, _ = context(root)
    images(root, cache)
    sources(root, cache)
    cache = Path(cache)
    # Build disposable complete-box detection evaluation data; original files are unchanged.
    for split in ("train", "val_select", "test"):
        for r in rows:
            if r["split"] != split or (
                r["animal"] and (r.get("incomplete_boxes", True) or not r["boxes"])
            ):
                continue
            for kind in ("images", "labels"):
                (cache / "detection" / kind / split).mkdir(parents=True, exist_ok=True)
            dest = cache / "detection/images" / split / (r["id"] + ".jpg")
            if not dest.exists():
                os.symlink(image_path(cache, r["id"]), dest)
            labels = []
            for b in r["boxes"]:
                x1, y1, x2, y2 = b["xyxy"]
                labels.append(f"0 {(x1+x2)/2} {(y1+y2)/2} {x2-x1} {y2-y1}")
            (cache / "detection/labels" / split / (r["id"] + ".txt")).write_text(
                "\n".join(labels)
            )
    save(
        cache / "detection/data.yaml",
        dict(
            path=str((cache / "detection").resolve()),
            train="images/train",
            val="images/val_select",
            test="images/test",
            names={0: "animal"},
        ),
    )
    # Disposable native M5 evaluation data preserve the original split names.
    release = read(root / "segmentation/reference_release.json")["mask_hashes"]
    for split, expected in [("train", 150), ("val_select", 40), ("test", 40)]:
        labels = sorted((root / "segmentation/labels" / split).glob("*.txt"))
        assert len(labels) == expected
        for kind in ("images", "labels"):
            (cache / "segmentation" / kind / split).mkdir(parents=True, exist_ok=True)
        for label in labels:
            assert sha(label) == release[label.stem]
            target = cache / "segmentation/images" / split / (label.stem + ".jpg")
            if not target.exists():
                os.symlink(image_path(cache, label.stem), target)
            shutil.copyfile(label, cache / "segmentation/labels" / split / label.name)
    save(
        cache / "segmentation/data.yaml",
        dict(
            path=str((cache / "segmentation").resolve()),
            train="images/train",
            val="images/val_select",
            test="images/test",
            names={0: "animal"},
        ),
    )
    shutil.copyfile(
        root / "segmentation/reference_release.json",
        out / "original_reference_release.json",
    )
    provenance = out / "source_data"
    provenance.mkdir(exist_ok=True)
    for name in ["manifest.json", "species.json"]:
        shutil.copyfile(
            root / "phase1_source/data/cct20_team_v2" / name, provenance / name
        )
    shutil.copytree(
        root / "segmentation/labels",
        provenance / "original_mask_labels",
        dirs_exist_ok=True,
    )

    save(
        out / "prepared.json",
        dict(
            manifest_sha256=MANIFEST_SHA,
            frames=len(rows),
            cache=str(cache),
            test_used_for_training=False,
        ),
    )


def validate_snapshot(root):
    root, out, _, _ = context(root)
    sources = {
        str(p.relative_to(Path(__file__).parent)): sha(p)
        for p in Path(__file__).parent.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".txt", ".csv", ".pdf", ".json")
        and "__pycache__" not in p.parts
    }
    path = out / "code_snapshot.json"
    if path.exists():
        assert (
            read(path) == sources
        ), "Code changed during this version; resume with the saved source ZIP or use a new version"
    else:
        save(path, sources)
        dest = out / "code"
        dest.mkdir(exist_ok=True)
        for p in Path(__file__).parent.iterdir():
            if p.is_file():
                shutil.copyfile(p, dest / p.name)
        shutil.copytree(
            Path(__file__).parent / "assets", dest / "assets", dirs_exist_ok=True
        )


def acceptance_only(root, cache):
    from artifacts import verdict

    verdict(root)


def package(root, cache):
    root, out, rows, _ = context(root)
    dest = out / "submission"
    dest.mkdir(exist_ok=True)
    policy = read(out / "policy.json")
    freeze = read(out / "freeze.json")
    assert sha(out / "policy.json") == freeze["policy_sha256"]
    required = {
        "Classifier training": out / "classifier/completed.json",
        "Classifier validation selection": out / "species_selection.json",
        "Operating sweep": out / "threshold_sweep.csv",
        "Mask validation comparison": out / "mask_selection.json",
        "Frozen policy": out / "freeze.json",
        "Detector historical AP": out / "historical_test/native_detector_ap.json",
        "Detector operating metrics": out / "historical_test/detection_metrics.json",
        "Species metrics": out / "historical_test/species_metrics.json",
        "Segmentation overlap": out / "historical_test/segmentation/metrics.json",
        "Operating mask AP": out
        / "historical_test/segmentation/operating_mask_ap.json",
        "Native guide mask AP": out
        / "historical_test/segmentation/native_mask_ap.json",
        "Detector size recall": out / "historical_test/detection_size_recall.csv",
        "Event counts": out / "historical_test/event_metrics.json",
        "CPU benchmark": out / "cpu_runtime/benchmark.json",
        "GPU benchmark": out / "gpu_runtime/benchmark.json",
        "ONNX measured parity": out / "onnx/parity.json",
        "HTTP executed request": out / "http/service_test.json",
        "HTTP all-stage demo": out / "http/demo_record.json",
        "Fresh interpreter replay": out / "http/clean_replay.json",
        "Acceptance decision": out / "decision.json",
        "Model card": dest / "MODEL_CARD.md",
        "Phase 2 report": dest / "Phase2_Report.pdf",
        "Final presentation": dest / "Final_Presentation.pdf",
        "Recorded fallback": dest / "Demo_Fallback.mp4",
        "Video inference": dest / "Detector_Video.mp4",
    }
    checks = []
    for name, path in required.items():
        present = path.is_file() and path.stat().st_size > 0
        verified = present
        detail = "File exists"
        if present and path.name == "parity.json":
            verified = read(path)["passed"]
            detail = "Measured detector equivalence"
        elif present and path.name == "service_test.json":
            verified = (
                read(path)["status"] == 200
                and read(path)["invalid_request_status"] == 400
            )
        elif present and path.name == "demo_record.json":
            verified = read(path)["coverage"]["all_stages_exercised"]
        elif present and path.suffix == ".pdf":
            from pypdf import PdfReader

            verified = len(PdfReader(path).pages) > 0
            detail = "PDF parses; visual team review remains separate"
        if present and path.suffix == ".json":
            identity = read(path).get("policy_sha256")
            if identity is not None:
                verified = verified and identity == freeze["policy_sha256"]
        checks.append(
            dict(
                requirement=name,
                present=present,
                verified=verified,
                evidence=str(path),
                detail=detail,
                sha256=sha(path) if present else "",
            )
        )
    coverage = read(out / "http/demo_record.json")["coverage"]["all_stages_exercised"]
    checks.append(
        dict(
            requirement="HTTP all model stages",
            present=True,
            verified=coverage,
            evidence=str(out / "http/demo_record.json"),
            detail="Actually exercised species and masks",
            sha256=sha(out / "http/demo_record.json"),
        )
    )
    csv_save(dest / "completion.csv", checks)
    save(
        dest / "audit_status.json",
        dict(
            measured_artifacts_complete=all(c["verified"] for c in checks),
            performance_decision=read(out / "decision.json")["performance_decision"],
            team_visual_review_complete=False,
            new_failure_analysis_human_review_complete=False,
            scope="Artifact execution/consistency audit; human sign-off and deployment criteria are separate",
            policy_sha256=freeze["policy_sha256"],
        ),
    )
    weights = out / "release_models"
    weights.mkdir(exist_ok=True)
    models = {}
    for name, spec in [
        ("detector", policy["detector"]),
        ("animal_classifier", policy["animal_classifier"]),
        ("species_classifier", policy["species_classifier"]),
        ("mask_guide", policy["mask"]["guide"]),
    ]:
        assert sha(spec["path"]) == spec["sha256"]
        target = weights / (spec["sha256"][:16] + ".pt")
        if not target.exists() or sha(target) != spec["sha256"]:
            shutil.copyfile(spec["path"], target)
        models[name] = dict(
            relative_path=str(target.relative_to(out)), sha256=sha(target)
        )
    # Public SAM weights are reproducibly downloaded rather than duplicated into every archive.
    if policy["mask"]["kind"] != "direct":
        models["sam"] = dict(
            sha256=policy["mask"]["sam"]["sha256"],
            url="https://github.com/ultralytics/assets/releases/download/v8.3.0/"
            + ("sam_b.pt" if policy["mask"]["kind"] == "sam_b" else "sam2.1_t.pt"),
        )
    save(
        out / "release_manifest.json",
        dict(
            policy_sha256=freeze["policy_sha256"],
            models=models,
            source_manifest_sha256=MANIFEST_SHA,
            original_root=str(root),
            instructions="Drive installation uses original_root paths. For a different machine restore models, original manifest/images/reference labels, install saved requirements, then rewrite policy paths and re-freeze as a relocated release; preserve the original policy digest in provenance.",
        ),
    )
    (dest / "README.txt").write_text(
        "Camera-Trap Project 5 — measured final-sprint release\n\nOpen Phase2_Report.pdf and Final_Presentation.pdf. Demo_Fallback.mp4 replays a real HTTP response. Detector_Video.mp4 uses an explicitly artificial image sequence.\n\nMetric outcomes are in acceptance.csv and decision.json; artifact completion is separate from unattended deployment readiness. Original Phase 1 artifacts are unchanged. Human visual review of the generated documents/video and new failure-analysis drafts remains required.\n\nThe ZIP includes learned checkpoints, source code, metrics and provenance. Public SAM weights are referenced by URL/SHA. Original images and reference data remain in Drive. No API keys or image metadata credentials are included.\n"
    )
    package_dir = root / "submission_packages"
    package_dir.mkdir(exist_ok=True)
    target = package_dir / "Final_Sprint_Evidence.zip"
    temp = target.with_suffix(".tmp")
    excluded = {"resume.pt"}
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=2) as z:
        for path in out.rglob("*"):
            if (
                not path.is_file()
                or path.name in excluded
                or "__pycache__" in path.parts
                or "detector_cache" in path.parts
                or path.suffix == ".log"
            ):
                continue
            if path.suffix == ".pt" and "release_models" not in path.parts:
                continue
            z.write(path, path.relative_to(out))
    temp.replace(target)
    save(
        package_dir / "Final_Sprint_Evidence_receipt.json",
        dict(
            path=str(target),
            sha256=sha(target),
            bytes=target.stat().st_size,
            policy_sha256=freeze["policy_sha256"],
            packaged_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    print("PACKAGE:", target, flush=True)
    print(
        "Artifact audit:",
        all(c["verified"] for c in checks),
        "Performance:",
        read(out / "decision.json")["performance_decision"],
        flush=True,
    )


STAGES = [
    "prepare",
    "train",
    "species",
    "detector",
    "masks",
    "test",
    "cpu",
    "onnx",
    "http",
    "video",
    "documents",
    "package",
]


def stage(name, root, cache):
    if name == "prepare":
        return prepare(root, cache)
    if name == "train":
        from train import train

        return train(root, cache)
    if name in ["species", "detector", "masks", "test", "cpu"]:
        import evaluate

        return {
            "species": evaluate.species_selection,
            "detector": evaluate.detector_selection,
            "masks": evaluate.mask_selection,
            "test": evaluate.historical_test,
            "cpu": evaluate.cpu_runtime,
        }[name](root, cache)
    if name == "onnx":
        import parity

        return parity.run(root, cache)
    if name == "http":
        import service

        return service.run(root, cache)
    if name == "video":
        import video

        return video.run(root, cache)
    if name == "documents":
        import artifacts

        return artifacts.build(root, cache)
    if name == "package":
        return package(root, cache)
    raise ValueError(name)


def run(root, cache):
    import torch
    import fcntl

    lock_name = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    lock = open("/tmp/final_sprint_" + lock_name + ".lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        raise RuntimeError(
            "A final-sprint run is already active (PID "
            + lock.read().strip()
            + "). Do not start overlapping training. Check its Drive logs."
        )
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()

    root, out, _, _ = context(root)
    assert (
        torch.cuda.is_available()
    ), "Select a Colab GPU before starting. Completed stages and checkpoints are kept in Drive."
    validate_snapshot(root)
    began = time.time()
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    state_path = out / "progress.json"
    state = (
        read(state_path)
        if state_path.exists()
        else dict(completed=[], failed=None, stage_seconds={})
    )
    for i, name in enumerate(STAGES, 1):
        # Local Colab cache can expire; verify/restore input pixels every session.
        if name in state["completed"] and name != "prepare":
            print(f"[{i}/{len(STAGES)}] REUSED {name}", flush=True)
            continue
        print(
            f"[{i}/{len(STAGES)}] START {name} — {datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )
        start = time.time()
        state["current"] = name
        state["failed"] = None
        save(state_path, state)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--cache",
            str(cache),
            "--stage",
            name,
        ]
        with (logs / (name + ".log")).open("a", buffering=1) as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
        if code:
            state["failed"] = name
            save(state_path, state)
            raise RuntimeError(
                f'{name} failed (exit {code}). Log: {logs/(name+".log")}. Correct the reported issue and rerun the same cell; completed stages are retained.'
            )
        if name not in state["completed"]:
            state["completed"].append(name)
        state["stage_seconds"][name] = time.time() - start
        state["current"] = None
        save(state_path, state)
        print(
            f'[{i}/{len(STAGES)}] SAVED {name} — {state["stage_seconds"][name]/60:.1f} min',
            flush=True,
        )
    save(
        out / "execution_receipt.json",
        dict(
            all_automated_stages_executed=True,
            total_session_seconds=time.time() - began,
            policy_sha256=sha(out / "policy.json"),
            human_review_claimed=False,
        ),
    )
    print("AUTOMATED RUN FINISHED. Review submission/ and acceptance.csv.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache", default="/content/final_sprint_cache")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = parser.parse_args()
    if args.stage == "all":
        run(args.root, args.cache)
    else:
        stage(args.stage, Path(args.root), Path(args.cache))
