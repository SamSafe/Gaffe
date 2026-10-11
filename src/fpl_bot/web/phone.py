"""Paired phone gateway: HTTPS LAN and an optional authenticated HTTPS tunnel."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import ipaddress
import os
import platform
import re
import secrets
import socket
import subprocess
import threading
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlsplit

import httpx
import qrcode
import qrcode.image.svg
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from fpl_bot.web.store import Store, read_json, timestamp, write_json

COOKIE = "gaffe_device"
LOGIN = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pair your phone · Gaffe</title><link rel="stylesheet" href="/assets/style.css"><main><div class="eyebrow">GAFFE · PHONE ACCESS</div><h1>Bring your dugout with you.</h1><section class="card"><div class="card-body"><p>On your desktop, open Settings → Phone access → Create pairing link.</p><p>Scan the QR code or open the link on this phone. Each link works once and expires after five minutes.</p></div></section></main></html>"""


def local_addresses() -> list[str]:
    addresses = {"127.0.0.1"}
    with suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("1.1.1.1", 80))  # Route lookup; no packet is sent.
        addresses.add(sock.getsockname()[0])
    return sorted(addresses)


class PhoneAccess:
    def __init__(self, store: Store, desktop_port: int = 8080):
        self.store, self.desktop_port = store, desktop_port
        self.private = store.web / "private"
        self.servers, self.threads = [], []
        self.tunnel = None
        self.lock = threading.RLock()
        self.state = "disabled"
        self.public_url = None
        self.addresses = local_addresses()
        self.message = "Enable phone access to create a paired connection."
        self.pairings = {}
        self.closed = False

    def status(self) -> dict:
        return {
            "status": self.state,
            "message": self.message,
            "public_url": self.public_url,
            "local_urls": [f"https://{ip}:8443" for ip in self.addresses if ip != "127.0.0.1"],
            "devices": len(self._sessions()),
            "pairing_minutes": 5,
            "helper_installed": (self.store.web / "bin" / "cloudflared").exists(),
        }

    def install_tunnel(self):
        """Fetch only the official platform binary and verify GitHub's SHA-256 digest."""
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
        if platform.system() != "Linux" or not architecture:
            raise ValueError("Automatic phone helper installation supports Linux amd64 and arm64.")
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            release = client.get(
                "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
            )
            release.raise_for_status()
            asset = next(
                a
                for a in release.json()["assets"]
                if a["name"] == f"cloudflared-linux-{architecture}"
            )
            checksum = asset.get("digest", "")
            url = asset["browser_download_url"]
            if not checksum.startswith("sha256:") or not url.startswith(
                "https://github.com/cloudflare/cloudflared/releases/download/"
            ):
                raise ValueError("The official phone helper checksum is unavailable.")
            folder = self.store.web / "bin"
            folder.mkdir(parents=True, exist_ok=True)
            staged = folder / "cloudflared.download"
            sha = hashlib.sha256()
            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with staged.open("wb") as stream:
                        for chunk in response.iter_bytes():
                            stream.write(chunk)
                            sha.update(chunk)
                if sha.hexdigest() != checksum.removeprefix("sha256:"):
                    raise ValueError("The phone helper checksum did not match.")
                os.chmod(staged, 0o750)
                staged.replace(folder / "cloudflared")
            finally:
                staged.unlink(missing_ok=True)

    def _sessions(self):
        data = read_json(self.private / "devices.json", {})
        now = dt.datetime.now(dt.UTC)
        return {k: v for k, v in data.items() if timestamp(v["expires_at"]) > now}

    def paired(self, raw: str | None, host: str = "testserver") -> bool:
        return bool(
            raw
            and self._sessions().get(hashlib.sha256(raw.encode()).hexdigest(), {}).get("host")
            == host
        )

    def pair_link(self) -> dict:
        if self.state not in {"ready", "local"}:
            raise ValueError("Phone access is still starting. Wait a moment and try again.")
        base = self.public_url or next(
            (f"https://{ip}:8443" for ip in self.addresses if ip != "127.0.0.1"), None
        )
        if not base:
            raise ValueError("No reachable network address is available.")
        nonce = secrets.token_urlsafe(24)
        expiry = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5)
        with self.lock:
            self.pairings = {k: v for k, v in self.pairings.items() if v > dt.datetime.now(dt.UTC)}
            self.pairings[hashlib.sha256(nonce.encode()).hexdigest()] = expiry
        link = f"{base}/pair/{nonce}"
        qr = qrcode.make(link, image_factory=qrcode.image.svg.SvgPathImage)
        stream = io.BytesIO()
        qr.save(stream)
        return {"url": link, "expires_at": expiry.isoformat(), "qr_svg": stream.getvalue().decode()}

    def consume(self, nonce: str, host: str = "testserver") -> str | None:
        key = hashlib.sha256(nonce.encode()).hexdigest()
        with self.lock:
            expiry = self.pairings.pop(key, None)
            if not expiry or expiry <= dt.datetime.now(dt.UTC):
                return None
            token = secrets.token_urlsafe(32)
            devices = self._sessions()
            devices[hashlib.sha256(token.encode()).hexdigest()] = {
                "host": host,
                "paired_at": dt.datetime.now(dt.UTC).isoformat(),
                "expires_at": (dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
            }
            write_json(self.private / "devices.json", devices)
            return token

    def revoke(self):
        with self.lock:
            self.pairings.clear()
            write_json(self.private / "devices.json", {})

    def _certificate(self):
        cert_path, key_path = self.private / "phone.crt", self.private / "phone.key"
        if cert_path.exists() and key_path.exists():
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            if cert.not_valid_after_utc > dt.datetime.now(dt.UTC) + dt.timedelta(days=7) and all(
                ipaddress.ip_address(ip) in sans.get_values_for_type(x509.IPAddress)
                for ip in self.addresses
            ):
                return cert_path, key_path
        self.private.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Gaffe local phone access")])
        now = dt.datetime.now(dt.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName("localhost")]
                    + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in self.addresses]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(key_path, 0o600)
        return cert_path, key_path

    def start(self):
        with self.lock:
            if self.state != "disabled":
                return
            self.closed = False
            self.state = "starting"
            self.message = "Starting your paired phone gateway…"
        thread = threading.Thread(target=self._start, daemon=True)
        self.threads.append(thread)
        thread.start()

    def _start(self):
        try:
            cert, key = self._certificate()
            for host, port, tls in [("0.0.0.0", 8443, True), ("127.0.0.1", 8444, False)]:
                server = uvicorn.Server(
                    uvicorn.Config(
                        self.gateway(),
                        host=host,
                        port=port,
                        access_log=False,
                        log_level="error",
                        **({"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if tls else {}),
                    )
                )
                self.servers.append(server)
                thread = threading.Thread(target=server.run, daemon=True)
                self.threads.append(thread)
                thread.start()
            self.state = "local"
            self.message = "Local HTTPS is ready. Connecting your public HTTPS phone link…"
            executable = self.store.web / "bin" / "cloudflared"
            if not executable.exists():
                self.message = (
                    "Local HTTPS is ready. Install the tunnel helper for access across networks."
                )
                return
            self.tunnel = subprocess.Popen(
                [str(executable), "tunnel", "--url", "http://127.0.0.1:8444", "--no-autoupdate"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            for line in self.tunnel.stderr:
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if match and not self.closed:
                    self.public_url = match.group()
                    self.state = "ready"
                    self.message = "Encrypted phone access is ready. Create a one-use pairing link."
            if not self.closed:
                self.public_url = None
                self.state = "local"
                self.message = "The tunnel stopped. Local HTTPS remains available; toggle phone access to reconnect."
        except (OSError, ValueError):
            self.state = "failed"
            self.message = "Could not start phone access. Check ports 8443 and 8444."

    def stop(self):
        self.closed = True
        for server in self.servers:
            server.should_exit = True
        if self.tunnel and self.tunnel.poll() is None:
            self.tunnel.terminate()
        for thread in self.threads:
            thread.join(timeout=3)
        self.servers = []
        self.threads = []
        self.tunnel = None
        self.public_url = None
        self.state = "disabled"
        self.message = "Phone access is disabled."

    def gateway(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            app.state.client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self.desktop_port}", timeout=30
            )
            yield
            await app.state.client.aclose()

        app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

        @app.middleware("http")
        async def boundary(request, call_next):
            host = urlsplit(f"//{request.headers.get('host', '')}").hostname
            allowed = set(self.addresses) | {"localhost", "testserver"}
            if self.public_url:
                allowed.add(urlsplit(self.public_url).hostname)
            if host not in allowed:
                return JSONResponse({"detail": "Unknown phone gateway host."}, status_code=400)
            response = await call_next(request)
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/pair/{nonce}")
        def pair(nonce: str, request: Request):
            token = self.consume(nonce, request.url.hostname)
            if not token:
                return HTMLResponse(
                    "Pairing link expired or already used. Create another on your desktop.",
                    status_code=401,
                )
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                COOKIE, token, max_age=30 * 86400, secure=True, httponly=True, samesite="strict"
            )
            return response

        @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy(path: str, request: Request):
            if path.startswith("/") or any(part in {".", ".."} for part in path.split("/")):
                return JSONResponse({"detail": "Invalid app path."}, status_code=400)
            if not self.paired(request.cookies.get(COOKIE), request.url.hostname):
                if path == "assets/style.css":
                    from fastapi.responses import FileResponse

                    from fpl_bot.web.app import ASSETS

                    return FileResponse(ASSETS / "style.css")
                return HTMLResponse(LOGIN, status_code=200 if not path else 401)
            origin = request.headers.get("origin")
            if (
                request.method not in {"GET", "HEAD"}
                and origin
                and urlsplit(origin).netloc != request.headers.get("host")
            ):
                return JSONResponse({"detail": "Use your paired app window."}, status_code=403)
            # Pairing/revocation remains a desktop-only operation.
            if path.startswith("api/access"):
                return JSONResponse(
                    {"detail": "Manage paired devices on your desktop."}, status_code=403
                )
            headers = {
                k: v
                for k, v in request.headers.items()
                if k
                in {"content-type", "accept", "x-gaffe-token", "if-none-match", "if-modified-since"}
            }
            headers["x-gaffe-device"] = "paired"
            try:
                result = await request.app.state.client.request(
                    request.method,
                    "/" + path,
                    params=request.query_params,
                    headers=headers,
                    content=await request.body(),
                )
            except httpx.HTTPError:
                return JSONResponse(
                    {"detail": "The desktop app is starting. Try again shortly."}, status_code=502
                )
            excluded = {"content-length", "transfer-encoding", "connection", "content-encoding"}
            return Response(
                result.content,
                status_code=result.status_code,
                headers={k: v for k, v in result.headers.items() if k not in excluded},
            )

        return app
