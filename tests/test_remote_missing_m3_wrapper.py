from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_sync_directory_preserves_contents_at_exact_remote_path(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "package"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "rsync-args.txt"
    for name, body in {
        "ssh": "#!/bin/sh\nexit 0\n",
        "rsync": '#!/bin/sh\nprintf "%s\\n" "$@" > "$RSYNC_ARGS_LOG"\n',
    }.items():
        executable = fake_bin / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "RSYNC_ARGS_LOG": str(log),
            "GCNET_REMOTE_HOST": "example",
            "GCNET_REMOTE_ROOT": "/remote/repo",
        }
    )
    relative_source = source.relative_to(tmp_path)
    completed = subprocess.run(
        [str(repo / "scripts/remote_missing_m3.sh"), "sync", str(relative_source)],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert arguments == [
        "-a",
        "package/",
        "example:/remote/repo/package/",
    ]
