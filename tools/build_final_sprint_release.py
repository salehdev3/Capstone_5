"""Build a deterministic source archive and, after commit, its pinned one-cell notebook."""

from pathlib import Path
import argparse, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/final_sprint"
DEST = ROOT / "releases/final_sprint"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(revision=None):
    DEST.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p
        for p in SOURCE.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "bootstrap_template.py"
    )
    archive = DEST / "Final_Sprint_Code.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for p in paths:
            info = zipfile.ZipInfo(
                str(p.relative_to(SOURCE)), date_time=(2026, 11, 20, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    sha = digest(archive)
    (DEST / "source_checksums.json").write_text(
        json.dumps(
            dict(
                archive_sha256=sha,
                files={str(p.relative_to(SOURCE)): digest(p) for p in paths},
            ),
            indent=2,
        )
        + "\n"
    )
    print("Source:", archive, "SHA256:", sha)
    if revision:
        assert len(revision) == 40 and all(
            c in "0123456789abcdef" for c in revision
        ), "Use a full immutable commit SHA"
        code = (
            (SOURCE / "bootstrap_template.py")
            .read_text()
            .replace("__COMMIT__", revision)
            .replace("__SHA256__", sha)
        )
        compile(code, "one_cell.py", "exec")
        notebook = dict(
            nbformat=4,
            nbformat_minor=5,
            metadata=dict(
                accelerator="GPU",
                kernelspec=dict(
                    display_name="Python 3", language="python", name="python3"
                ),
                language_info=dict(name="python", version="3.13"),
            ),
            cells=[
                dict(
                    cell_type="markdown",
                    id="overview",
                    metadata={},
                    source=[
                        "# Phase 2 — Final Sprint\n",
                        "Run the single cell below on a Colab GPU with the Drive account containing `CameraTrap_Phase2`.\n",
                        "\n",
                        "Training, validation selection, historical evaluation, HTTP/ONNX, CPU/GPU benchmarks and submission drafts run in order. Progress and checkpoints are saved to Drive; rerun the same cell to resume.\n",
                        "\n",
                        "Outputs: `phase2_results/final_sprint_v1/submission/` and `submission_packages/Final_Sprint_Evidence.zip`. Review the measured results and generated files before submission.\n",
                    ],
                ),
                dict(
                    cell_type="code",
                    id="run-everything",
                    metadata={},
                    execution_count=None,
                    outputs=[],
                    source=code.splitlines(keepends=True),
                ),
            ],
        )
        path = ROOT / "notebooks/Phase2_Final_Sprint_One_Cell.ipynb"
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        (DEST / "Run_Final_Sprint_One_Cell.py").write_text(code)
        print("Notebook:", path)
    return sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    args = parser.parse_args()
    build(args.revision)
