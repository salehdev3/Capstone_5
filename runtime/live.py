"""Run the frozen integrated release through its local HTTP batch interface."""
from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile
import time

import requests
from PIL import Image, ImageDraw
from common import read, save, sha
from deployment import GuidedPipeline
from http_service import start_service
from runtime import DEMO_IDS

RUN = "phase2_results/integrated_species_cv/20261120T224149142114Z"


def initialize(root):
    release = root / RUN
    policy = release / "policy.json"
    decision = read(release / "decision.json")
    if sha(policy) != decision["policy_sha256"]:
        raise ValueError("The policy differs from the integrated evaluation.")
    cache = Path(tempfile.mkdtemp(prefix="camera_demo_"))
    return GuidedPipeline(root, cache, policy), decision


def cases(root):
    rows = read(root / "phase1_source/data/cct20_team_v2/manifest.json")
    lookup = {r["id"]: r for r in rows}
    return [(root / "images" / (ident + ".jpg"), lookup[ident]) for ident in DEMO_IDS]



def team_test_cases(root):
    rows = read(root / "phase1_source/data/cct20_team_v2/manifest.json")
    selected = [r for r in rows if r.get("split") == "test" and r.get("source") == "team_collected"]
    if not selected:
        raise ValueError("No team_collected test images found in the manifest.")
    batch = [(root / "images" / (r["id"] + ".jpg"), r) for r in selected]
    missing = [str(path) for path, _ in batch if not path.is_file()]
    if missing:
        raise FileNotFoundError("Team test images missing from Drive:\n" + "\n".join(missing))
    return batch


def annotate(path, frame):
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for obj in frame["objects"]:
        box = obj["box"]
        x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
        draw.rectangle((x1,y1,x2,y2), outline="orange", width=3)
        draw.text((x1,max(0,y1-16)), f"{obj['species']} {obj['confidence']:.2f}", fill="yellow",stroke_width=1,stroke_fill="black")
        points = [(x*w,y*h) for x,y in obj.get("polygon_normalized",[])]
        if len(points)>=3:
            draw.line(points+[points[0]],fill="lime",width=3)
    return image


def execute(pipeline, batch, destination):
    server, thread, base, instance = start_service(pipeline)
    try:
        records = [dict(id=r["id"],event=r["event"],timestamp=r.get("timestamp")) for _,r in batch]
        images = [("images",(path.name,path.read_bytes(),"application/octet-stream")) for path,_ in batch]
        start = time.perf_counter()
        response = requests.post(base+"/batch", files=images, data={"metadata":json.dumps(records)},timeout=(10,600))
        response.raise_for_status()
        elapsed = time.perf_counter()-start
        result = response.json()
        assert [f["id"] for f in result["frames"]]==[r["id"] for r in records]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    dest = destination / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest.mkdir(parents=True)
    save(dest/"service_test.json",dict(policy_sha256=pipeline.policy_sha256,elapsed_seconds=elapsed,
                                       http_status=response.status_code,instance=instance,result=result))
    import imageio_ffmpeg
    import subprocess
    video = dest/"Demo_Fallback.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        for index,((path,_),frame) in enumerate(zip(batch,result["frames"])):
            image = annotate(path,frame)
            image.thumbnail((960,640))
            canvas = Image.new("RGB",(960,720),"#162b40")
            canvas.paste(image,((960-image.width)//2,12))
            draw = ImageDraw.Draw(canvas)
            draw.text((20,665),f"Frame {index+1}/{len(batch)}   |   Animals: {frame['count']}   |   Inference: {frame['timing']['total']:.2f} s",fill="white")
            draw.text((20,690),"Orange: detection    Green: segmentation    Yellow: species and confidence",fill="white")
            canvas.save(Path(tmp)/f"{index:04}.png")
            canvas.save(dest/f"frame_{index+1:02}.jpg")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-loglevel","error","-framerate","1/3","-i",str(Path(tmp)/"%04d.png"),
                        "-vf","fps=15","-pix_fmt","yuv420p","-c:v","libx264",str(video)],check=True)
    return dest,result


def video_cases(path, directory, limit=5):
    """Sample one frame per second for a short event demonstration."""
    import cv2
    import math

    directory.mkdir(parents=True, exist_ok=True)
    reader = cv2.VideoCapture(str(path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not reader.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Select a readable video with a known frame rate.")
        batch = []
        for second in range(limit):
            reader.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
            ok, frame = reader.read()
            if not ok:
                break
            ident = f"video-{second:03}"
            image = directory / (ident + ".jpg")
            if not cv2.imwrite(str(image), frame):
                raise OSError(f"Could not save {image}")
            batch.append((image, dict(id=ident, event=path.stem, timestamp=f"{second}s")))
        if not batch:
            raise ValueError("The video contains no readable frames.")
        return batch
    finally:
        reader.release()
