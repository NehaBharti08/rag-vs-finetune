class TestTheFrozenSetIsDistributed:
    """A freeze over a file nobody else has is not a freeze.

    `configs/eval/frozen.lock` hashes `data/eval/gold.jsonl`. That file was
    gitignored, so a clean checkout failed the harness check with
    "removed data/eval/gold.jsonl" -- found by the first end-to-end
    reproduction. Every eval number in this repo was unreproducible by a third
    party, and the pre-registration was unverifiable.
    """

    def test_every_frozen_path_is_tracked_by_git(self) -> None:
        import subprocess

        from ragft.eval.frozen import read_lock
        from ragft.settings import REPO_ROOT

        lock = read_lock()
        assert lock.get("frozen"), "harness must be frozen"

        untracked = []
        for path in lock["files"]:
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", path],
                cwd=REPO_ROOT,
                capture_output=True,
            )
            if result.returncode != 0:
                untracked.append(path)

        assert not untracked, (
            "frozen.lock hashes files that are NOT committed: "
            f"{untracked}. Anyone cloning this repo cannot verify the frozen "
            "harness, so the pre-registration claim is unverifiable and no eval "
            "number can be reproduced."
        )
