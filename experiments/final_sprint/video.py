"""Record an actual HTTP execution replay and executed detector inference on a labelled image sequence."""

from common import *
from runtime import DEMO_IDS
import numpy as np, cv2, subprocess


def run(root, cache):
    import imageio_ffmpeg
    from PIL import ImageDraw, ImageFont
    from ultralytics import YOLO

    root, out, rows, _ = context(root)
    dest = out / "submission"
    dest.mkdir(exist_ok=True)
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    width, height = 1024, 768
    fps = 10

    def writer(path):
        return subprocess.Popen(
            [
                encoder,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def emit(proc, im, seconds):
        raw = np.asarray(im, dtype=np.uint8).tobytes()
        for _ in range(round(seconds * fps)):
            proc.stdin.write(raw)

    def finish(proc):
        proc.stdin.close()
        assert proc.wait(timeout=90) == 0, "Video encoding failed"

    def card(title, lines, source=None):
        im = Image.new("RGB", (width, height), "#102a3a")
        d = ImageDraw.Draw(im)
        fontpath = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = (
            ImageFont.truetype(str(fontpath), 22)
            if fontpath.exists()
            else ImageFont.load_default()
        )
        small = ImageFont.truetype(str(fontpath), 16) if fontpath.exists() else font
        d.text((28, 18), title, fill="white", font=font)
        if source:
            with Image.open(source) as pic:
                pic = pic.convert("RGB")
                pic.thumbnail((968, 568))
                im.paste(pic, ((width - pic.width) // 2, 76))
        for i, line in enumerate(lines):
            d.text((28, 660 + i * 26), line[:110], fill="white", font=small)
        return im

    policy = read(out / "policy.json")
    http = read(out / "http/service_test.json")
    record = read(out / "http/demo_record.json")
    proc = writer(dest / "Demo_Fallback.mp4")
    emit(
        proc,
        card(
            "Camera-Trap Event Triage — Recorded Execution Replay",
            [
                "Real local HTTP POST /batch, status 200.",
                "Five fixed validation cases; this is a response replay, not a new accuracy evaluation.",
            ],
        ),
        5,
    )
    for f in http["result"]["frames"]:
        lines = [
            f"Frame {f['id']} | detections {f['count']} | matched masks {f['segmentation']['matched_masks']}",
            f"Measured complete inference: {f['timing']['total']:.3f} s | yellow: species | green: predicted mask",
        ]
        emit(
            proc,
            card(
                "HTTP response: animal detection, species and mask",
                lines,
                out / "http" / (f["id"] + ".png"),
            ),
            8,
        )
    emit(
        proc,
        card(
            "Event summary",
            [
                f"Returned events: {len(http['result']['events'])}; all-stage exercise: {record['coverage']['all_stages_exercised']}",
                "The live notebook can execute the same HTTP request from a fresh interpreter.",
            ],
        ),
        5,
    )
    finish(proc)
    # Explicit image-sequence input and actual detector inference on its decoded frames.
    source = dest / "Detector_Input_Sequence.mp4"
    proc = writer(source)
    for ident in DEMO_IDS:
        emit(
            proc,
            card(
                "Validation image sequence — no natural motion",
                [
                    "Artificial sequence of five project images; demonstration only.",
                    "Detector executes on the decoded sequence; metrics are calculated on original images.",
                ],
                image_path(cache, ident),
            ),
            1,
        )
    finish(proc)
    detector = YOLO(policy["detector"]["path"])
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened()
    proc = writer(dest / "Detector_Video.mp4")
    frames = 0
    last = None
    import torch

    while True:
        okay, im = cap.read()
        if not okay:
            break
        # Run at the beginning of each static-image segment and reuse its result for the identical frames.
        if frames % fps == 0:
            last = detector.predict(
                im,
                conf=policy["detector"]["conf"],
                iou=PLAN["nms"],
                max_det=PLAN["max_boxes"],
                imgsz=policy["detector"]["size"],
                rect=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0].plot()
        proc.stdin.write(cv2.cvtColor(last, cv2.COLOR_BGR2RGB).tobytes())
        frames += 1
    cap.release()
    finish(proc)
    save(
        dest / "video_provenance.json",
        dict(
            http_response_sha256=sha(out / "http/service_test.json"),
            policy_sha256=sha(out / "policy.json"),
            fallback_kind="Rendered replay of the actually executed HTTP response, not a recording of an unexecuted UI",
            detector_video_kind="Actual detector inference on a decoded artificial validation-image sequence; no natural-video performance claim",
            decoded_frames=frames,
            inference_frames=(frames + fps - 1) // fps,
            files={
                name: sha(dest / name)
                for name in [
                    "Demo_Fallback.mp4",
                    "Detector_Input_Sequence.mp4",
                    "Detector_Video.mp4",
                ]
            },
        ),
    )
    print("Videos saved:", dest, flush=True)
