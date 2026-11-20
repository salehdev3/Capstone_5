"""Optional authenticated backup; never record a success before the artifact commits."""

from common import *
import os


def run(root):
    import wandb

    root, out, _, _ = context(root)
    archive = root / "submission_packages/Final_Sprint_Evidence.zip"
    receipt = out / "submission/wandb_link.json"
    assert archive.is_file()
    if receipt.exists():
        old = read(receipt)
        if old["archive_sha256"] == sha(archive) and old["status"] == "COMMITTED":
            print("Same archive already backed up:", old["url"], flush=True)
            return
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    with wandb.init(
        project="camera-trap-phase2",
        job_type="final-sprint-backup",
        config=dict(
            version=PLAN["version"],
            policy_sha256=sha(out / "policy.json"),
            original_phase1_modified=False,
        ),
    ) as run:
        artifact = wandb.Artifact(
            "final-sprint-evidence",
            type="evaluation",
            metadata=dict(
                archive_sha256=sha(archive), policy_sha256=sha(out / "policy.json")
            ),
        )
        artifact.add_file(str(archive))
        logged = run.log_artifact(artifact)
        logged.wait(timeout=180)
        assert logged.state == "COMMITTED"
        save(
            receipt,
            dict(
                url=run.url,
                artifact_name=logged.qualified_name,
                artifact_digest=logged.digest,
                archive_sha256=sha(archive),
                status="COMMITTED",
            ),
        )
        print("BACKUP COMMITTED:", run.url, flush=True)


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
