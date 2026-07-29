from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast
from urllib.parse import parse_qs, unquote, urlparse

import requests
from PIL import Image

from texas_mushrooms.pipeline.photo_assets import index_local_images, match_local_file

try:
    # Keep UA consistent with the scraper.
    from texas_mushrooms.scrape.core import USER_AGENT
except Exception:  # pragma: no cover
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


ALLOWED_NETLOCS: Final[set[str]] = {
    "www.texasmushrooms.org",
    "texasmushrooms.org",
    "www.asergeev.com",
    "asergeev.com",
}


# Bounds for the optional `w` (max-width) thumbnail parameter.
_MIN_THUMB_W: Final[int] = 32
_MAX_THUMB_W: Final[int] = 1600


def _parse_width(qs: dict[str, list[str]]) -> int | None:
    raw = (qs.get("w") or [""])[0].strip()
    if not raw:
        return None
    try:
        w = int(raw)
    except ValueError:
        return None
    return max(_MIN_THUMB_W, min(_MAX_THUMB_W, w))


def _resize_jpeg(data: bytes, max_w: int) -> bytes:
    """Downscale image bytes so width <= ``max_w``; return JPEG bytes.

    Returns the original bytes unchanged if the image can't be decoded or is
    already narrower than the target (avoids upscaling).
    """
    try:
        with Image.open(io.BytesIO(data)) as src:
            im = src.convert("RGB")
            if im.width > max_w:
                h = round(im.height * (max_w / im.width))
                im = im.resize((max_w, h), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=82, optimize=True)
            return out.getvalue()
    except Exception:
        return data


def _safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "texas-mushrooms-web-proxy/1.0"

    @property
    def images_root(self) -> Path:
        return cast(Path, self.server.images_root)  # type: ignore[attr-defined]

    @property
    def thumb_root(self) -> Path:
        return cast(Path, self.server.thumb_root)  # type: ignore[attr-defined]

    def _local_copy(self, date: str, url: str) -> Path | None:
        """Resolve a photo URL to its already-downloaded file, if we have it.

        The scrape keeps every image under ``data/raw/images/<date>/``, so a
        hover preview almost never needs the network. The date disambiguates:
        basenames like ``12b.jpg`` repeat across days.
        """
        if not date:
            return None
        index = cast(
            "dict[str, list[Path]]",
            self.server.local_index,  # type: ignore[attr-defined]
        )
        return match_local_file(date, url, index)

    def _send_text(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_text(HTTPStatus.OK, "ok")
            return

        if parsed.path == "/proxy":
            self._handle_proxy(parsed.query)
            return

        # Serve local images from data/raw/images/<date>/<filename>
        self._handle_static(parsed.path)

    def _handle_static(self, path: str) -> None:
        rel = unquote(path).lstrip("/")
        if not rel:
            self._send_text(HTTPStatus.NOT_FOUND, "not found")
            return

        file_path = _safe_join(self.images_root, rel)
        if file_path is None or not file_path.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, "not found")
            return

        ctype, _ = mimetypes.guess_type(str(file_path))
        if not ctype:
            ctype = "application/octet-stream"

        try:
            data = file_path.read_bytes()
        except OSError:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, "failed to read file")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _send_jpeg(self, data: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _handle_proxy(self, query: str) -> None:
        qs = parse_qs(query)
        url = (qs.get("url") or qs.get("u") or [""])[0]
        ref = (qs.get("ref") or qs.get("r") or [""])[0]
        date = (qs.get("date") or qs.get("d") or [""])[0].strip()
        width = _parse_width(qs)

        url = url.strip()
        ref = ref.strip()

        if not url:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing url")
            return

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            self._send_text(HTTPStatus.BAD_REQUEST, "invalid url")
            return

        if parsed.netloc not in ALLOWED_NETLOCS:
            self._send_text(HTTPStatus.FORBIDDEN, "url host not allowed")
            return

        # Thumbnail path: serve a cached, downscaled JPEG when `w` is requested.
        cache_path: Path | None = None
        if width is not None:
            digest = hashlib.sha1(f"{width}|{url}".encode()).hexdigest()
            cache_path = self.thumb_root / f"{digest}.jpg"
            if cache_path.is_file():
                with contextlib.suppress(OSError):
                    self._send_jpeg(cache_path.read_bytes())
                    return

        # Prefer the copy we already scraped. Every photo in the display window
        # is on disk, so this turns a multi-second upstream round trip into a
        # local read -- and sends the source site no traffic at all.
        local = self._local_copy(date, url)
        if local is not None:
            try:
                raw = local.read_bytes()
            except OSError:
                raw = b""
            if raw:
                if width is None:
                    self._send_jpeg(raw)
                    return
                thumb = _resize_jpeg(raw, width)
                if cache_path is not None:
                    with contextlib.suppress(OSError):
                        cache_path.write_bytes(thumb)
                self._send_jpeg(thumb)
                return

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }

        if ref:
            ref_parsed = urlparse(ref)
            if ref_parsed.scheme in {"http", "https"}:
                headers["Referer"] = ref

        try:
            resp = requests.get(
                url, headers=headers, stream=True, timeout=30, allow_redirects=True
            )
        except requests.RequestException:
            self._send_text(HTTPStatus.BAD_GATEWAY, "upstream fetch failed")
            return

        if resp.status_code >= 400:
            self._send_text(
                HTTPStatus.BAD_GATEWAY, f"upstream status {resp.status_code}"
            )
            return

        # Thumbnail request: buffer, downscale, cache, and serve as JPEG.
        if width is not None:
            try:
                raw = resp.content
            finally:
                resp.close()
            thumb = _resize_jpeg(raw, width)
            if cache_path is not None:
                with contextlib.suppress(OSError):
                    cache_path.write_bytes(thumb)
            self._send_jpeg(thumb)
            return

        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        content_length = resp.headers.get("Content-Length")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if content_length:
            self.send_header("Content-Length", content_length)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()

        try:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    self.wfile.write(chunk)
        finally:
            resp.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve local images and proxy remote images for the web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--images-root",
        default=None,
        help="Directory containing downloaded images (defaults to data/raw/images).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    images_root = (
        Path(args.images_root)
        if args.images_root
        else (repo_root / "data" / "raw" / "images")
    )

    if not images_root.exists():
        raise SystemExit(f"images root not found: {images_root}")

    # On-disk cache for downscaled thumbnails (created lazily as requests land).
    thumb_root = repo_root / "data" / "processed" / "proxy_thumbs"
    thumb_root.mkdir(parents=True, exist_ok=True)

    # Ensure mime types are known on Windows.
    mimetypes.init()

    # Index the scrape once at startup so /proxy can answer from disk. Disk
    # names carry an on-page ordinal prefix the photo URL doesn't, so a lookup
    # table is cheaper than probing the filesystem per request.
    local_index = index_local_images(images_root)
    local_count = sum(len(v) for v in local_index.values())

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.images_root = images_root  # type: ignore[attr-defined]
    httpd.thumb_root = thumb_root  # type: ignore[attr-defined]
    httpd.local_index = local_index  # type: ignore[attr-defined]

    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Local images root: {images_root}")
    print("- Local images:   /YYYY-MM-DD/<filename>.jpg")
    print(
        "- Remote proxy:   /proxy?url=<photo_url>&ref=<page_url>"
        "[&date=YYYY-MM-DD][&w=<max_width>]"
    )
    print(f"- Thumb cache:    {thumb_root}")
    print(f"- Indexed {local_count} local images across {len(local_index)} days")
    print("  (pass &date= to serve from disk instead of fetching upstream)")

    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()


if __name__ == "__main__":
    main()
