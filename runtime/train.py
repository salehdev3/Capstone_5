from common import *
import torch, cv2, time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Crops(Dataset):
    def __init__(self, records, cache, augment=False):
        self.records = records
        self.cache = cache
        self.transforms = transforms.Compose(
            (
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.15, 0.15, 0.1, 0.01),
                    transforms.RandomGrayscale(0.05),
                ]
                if augment
                else []
            )
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            + ([transforms.RandomErasing(p=0.1, scale=(0.01, 0.08))] if augment else [])
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        im = cv2.imread(str(image_path(self.cache, r["id"])))
        assert im is not None
        return self.transforms(Image.fromarray(crop(im, r["box"]))), r["label"]


def samples(rows, classes, split):
    result = []
    for r in rows:
        if r["split"] != split:
            continue
        for b in r["boxes"]:
            if b["species"] in classes:
                result.append(
                    dict(id=r["id"], box=b["xyxy"], label=classes.index(b["species"]))
                )
    return result


def infer(net, loader, device):
    net.eval()
    zs = []
    ys = []
    with torch.inference_mode():
        for x, y in loader:
            zs.append(net(x.to(device)).float().cpu().numpy())
            ys.extend(y.tolist())
    return np.concatenate(zs), np.asarray(ys)


def prepare(root, cache):
    root, run, rows, classes = context(root)
    cache = Path(cache)
    (cache / "images").mkdir(parents=True, exist_ok=True)
    audit = []
    for i, r in enumerate(rows, 1):
        src = root / "images" / (r["id"] + ".jpg")
        local = Path("/content/phase2_v3_cache/images") / src.name
        if local.exists():
            src = local
        assert src.exists(), str(src)
        target = image_path(cache, r["id"])
        if not target.exists():
            shutil_copy(src, target)
        with Image.open(target) as im:
            rgb = im.convert("RGB")
            assert (
                rgb.size == (r["width"], r["height"])
                and hashlib.sha256(rgb.tobytes()).hexdigest() == r["pixel_sha256"]
            ), r["id"]
        audit.append(dict(id=r["id"], split=r["split"], sha256=sha(target)))
        if i % 100 == 0:
            print("Image verification", i, "/", len(rows), flush=True)
    save(run / "image_audit.json", audit)
    # Backgrounds come only from source-labelled empty training/calibration frames.
    for split in ("train", "val_calibrate"):
        records = samples(rows, classes, split)
        windows = [
            [0, 0, 1, 1],
            [0, 0, 0.55, 0.55],
            [0.45, 0.45, 1, 1],
            [0.2, 0.2, 0.8, 0.8],
        ]
        for r in rows:
            if r["split"] == split and not r["animal"]:
                records.extend(dict(id=r["id"], box=b, label=14) for b in windows)
        save(run / (split + "_samples.json"), records)
    save(
        run / "data_support.json",
        dict(
            classes=classes,
            training_counts=np.bincount(
                [r["label"] for r in read(run / "train_samples.json")], minlength=15
            ).tolist(),
            validation_counts=np.bincount(
                [r["label"] for r in samples(rows, classes, "val_select")], minlength=14
            ).tolist(),
            background_label=14,
            test_used_for_training=False,
        ),
    )


def shutil_copy(a, b):
    import shutil

    shutil.copyfile(a, b)


def train(root, cache):
    root, run, rows, classes = context(root)
    assert (
        torch.cuda.is_available()
    ), "GPU required. Select Colab GPU and rerun the same cell."
    seed()
    dest = run / "classifier"
    dest.mkdir(exist_ok=True)
    if (dest / "completed.json").exists():
        assert (
            sha(dest / "best.pt") == read(dest / "completed.json")["checkpoint_sha256"]
        )
        return
    device = "cuda"
    net = model(pretrained=not (dest / "resume.pt").exists()).to(device)
    for p in net.features.parameters():
        p.requires_grad = False
    selected = list(net.features[-3:].parameters())
    optimizer = torch.optim.AdamW(
        [
            dict(params=selected, lr=2e-5),
            dict(params=net.classifier.parameters(), lr=2e-4),
        ],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, PLAN["epochs"], eta_min=1e-6
    )
    train_data = Crops(read(run / "train_samples.json"), cache, True)
    loader = DataLoader(
        train_data,
        batch_size=PLAN["batch"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid = DataLoader(
        Crops(samples(rows, classes, "val_select"), cache), batch_size=48, num_workers=2
    )
    probe = DataLoader(
        Crops(random.Random(42).sample(samples(rows, classes, "train"), 512), cache),
        batch_size=48,
        num_workers=2,
    )
    counts = np.bincount([r["label"] for r in train_data.records], minlength=15)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = np.clip(weights / weights.mean(), 0.5, 2.5)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best = -1.0
    bad = 0
    start = 0
    if (dest / "resume.pt").exists():
        state = torch.load(dest / "resume.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best = state["best"]
        bad = state["bad"]
        start = state["epoch"] + 1
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    for epoch in range(start, PLAN["epochs"]):
        if bad >= PLAN["patience"]:
            break
        for p in selected:
            p.requires_grad = epoch >= 2
        net.train()
        net.features.eval()
        if epoch >= 2:
            net.features[-3:].train()
        began = time.perf_counter()
        total = 0.0
        seen = 0
        for step, (x, y) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(net(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
            if step == 1 or step % 25 == 0 or step == len(loader):
                elapsed = time.perf_counter() - began
                remaining = elapsed / step * (len(loader) - step)
                print(
                    f'Epoch {epoch+1}/{PLAN["epochs"]} | batch {step}/{len(loader)} | training-pass ETA {remaining/60:.1f} min',
                    flush=True,
                )
        scheduler.step()
        z, y = infer(net, valid, device)
        m = metrics(z[:, :14], y)
        pz, py = infer(net, probe, device)
        pm = metrics(pz[:, :14], py)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=total / seen,
                train_subset_accuracy=pm["accuracy"],
                val_accuracy=m["accuracy"],
                val_macro_f1=m["macro_f1"],
                accuracy_gap=pm["accuracy"] - m["accuracy"],
                seconds=time.perf_counter() - began,
            )
        )
        print("Classifier", history[-1], flush=True)
        if m["macro_f1"] > best + 1e-4:
            best = m["macro_f1"]
            bad = 0
            torch.save(
                dict(
                    architecture="convnext_tiny_bg15",
                    classes=classes,
                    state_dict={
                        k: v.detach().cpu() for k, v in net.state_dict().items()
                    },
                    epoch=epoch + 1,
                    validation=m,
                    version=PLAN["version"],
                ),
                dest / "best.tmp",
            )
            (dest / "best.tmp").replace(dest / "best.pt")
        else:
            bad += 1
        state = dict(
            model=net.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            best=best,
            bad=bad,
            history=history,
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
        )
        torch.save(state, dest / "resume.tmp")
        (dest / "resume.tmp").replace(dest / "resume.pt")
        csv_save(dest / "history.csv", history)
    net.load_state_dict(
        torch.load(dest / "best.pt", map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    cal = DataLoader(
        Crops(read(run / "val_calibrate_samples.json"), cache),
        batch_size=48,
        num_workers=2,
    )
    cz, cy = infer(net, cal, device)
    animal = cy < 14
    species_t = temperature(cz[animal, :14], cy[animal])
    full_t = temperature(cz, cy)
    z, y = infer(net, valid, device)
    save(
        dest / "policy.json",
        dict(
            species_temperature=species_t,
            animal_temperature=full_t,
            calibration_split="val_calibrate",
            checkpoint_sha256=sha(dest / "best.pt"),
            species_review_confidence=0.6,
        ),
    )
    save(dest / "validation.json", metrics(z[:, :14], y, species_t))
    save(
        dest / "completed.json",
        dict(
            checkpoint_sha256=sha(dest / "best.pt"),
            epochs=len(history),
            selection="val_select all-14 macro-F1",
            calibration="val_calibrate only",
            test_used=False,
        ),
    )
    np.savez_compressed(dest / "validation_logits.npz", logits=z, labels=y)
