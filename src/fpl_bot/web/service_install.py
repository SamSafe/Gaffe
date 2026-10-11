"""Install a per-user service without embedding credentials or requiring root."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def install(root: Path, enable: bool = True) -> Path:
    if not shutil.which("systemctl"):
        raise ValueError("Persistent service installation requires systemd on Linux.")
    root = root.resolve()

    def quoted(value):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'

    working_directory = str(root).replace("\\", "\\\\").replace("%", "%%")
    working_directory = working_directory.replace(" ", "\\x20").replace("\n", "\\n")

    body = f"""[Unit]
Description=Gaffe FPL dashboard and scheduled decisions
After=network-online.target

[Service]
Type=simple
WorkingDirectory={working_directory}
ExecStart={quoted(Path(sys.executable).absolute())} -m fpl_bot.web.launch
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
KillMode=mixed
NoNewPrivileges=true
UMask=0077
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    path = Path.home() / ".config/systemd/user/gaffe.service"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if enable:
        subprocess.run(["systemctl", "--user", "enable", "--now", "gaffe.service"], check=True)
        subprocess.run(["systemctl", "--user", "is-active", "--quiet", "gaffe.service"], check=True)
    return path
