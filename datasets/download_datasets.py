# download_datasets.py
"""
Dataset Download/Organize Utility for Indian Speech Validation.

WHAT THIS DOES:
- Provides helpers to fetch public corpora from OpenSLR, Common Voice (via HuggingFace), or HF datasets.
- Verifies basic structure and (optionally) transcript availability.
- Adds robust logging and "what to do if missing dependency" messages.

REQUIREMENTS:
- requests, tqdm (mandatory)
- Optional: datasets (HuggingFace), beautifulsoup4 (for OpenSLR index scraping)

SCIENTIFIC RIGOR:
- Log exact dataset versions/snapshots used.
- Keep a manifest.json with SHA256s when possible for reproducibility across runs.

USAGE:
    python download_datasets.py \
        --output_dir ./datasets \
        --languages hi,ta,te \
        --fetch common_voice,openslr \
        --split train,validation,test
"""

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# Optional deps
try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    _BS4_AVAILABLE = False


class DatasetDownloader:
    def __init__(self, output_dir: str = "./datasets", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (ASR-Downloader)"})

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # --------------------------
    # Generic helpers
    # --------------------------
    def _download_stream(self, url: str, dest: Path, desc: str = "Downloading") -> bool:
        try:
            self.log(f"GET {url}")
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f, tqdm(
                    total=total, unit="iB", unit_scale=True, desc=desc
                ) as bar:
                    for chunk in r.itercontent(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return True
        except Exception as e:
            self.log(f"❌ Download failed: {e}")
            return False

    def _extract(self, archive: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if str(archive).endswith(".zip"):
            with zipfile.ZipFile(archive, "r") as z:
                z.extractall(dest_dir)
        elif str(archive).endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive, "r:*") as t:
                t.extractall(dest_dir)
        else:
            self.log(f"Skipping extraction (unknown format): {archive}")

    # --------------------------
    # Common Voice via HF
    # --------------------------
    def fetch_common_voice(self, languages: List[str], split: List[str]) -> None:
        """
        Pulls Common Voice (latest) via HuggingFace datasets.
        NOTE: Requires `datasets` package and internet.
        """
        if not _HF_AVAILABLE:
            self.log("⚠️  'datasets' not installed. pip install datasets")
            return
        for lang in languages:
            for sp in split:
                self.log(f"Fetching Common Voice: lang={lang}, split={sp}")
                try:
                    ds = load_dataset("mozilla-foundation/common_voice_17_0", lang, split=sp)
                except Exception:
                    # fallback to 16 or 15 by trying a small list
                    for ver in ["16_1", "16_0", "15_0"]:
                        try:
                            ds = load_dataset(f"mozilla-foundation/common_voice_{ver}", lang, split=sp)
                            break
                        except Exception:
                            ds = None
                    if ds is None:
                        self.log(f"❌ Could not load Common Voice for {lang}/{sp}")
                        continue

                # Save a simple manifest (paths and text)
                out_dir = self.output_dir / "common_voice" / lang / sp
                out_dir.mkdir(parents=True, exist_ok=True)
                manifest = []
                for i in range(len(ds)):
                    row = ds[i]
                    path = row.get("path") or row.get("audio", {}).get("path")
                    text = row.get("sentence") or row.get("text")
                    if path is None:
                        continue
                    manifest.append({"path": str(path), "text": text})
                with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                self.log(f"✓ Saved manifest: {out_dir/'manifest.json'} (N={len(manifest)})")

    # --------------------------
    # OpenSLR (index scraping)
    # --------------------------
    def fetch_openslr(self, slr_ids: List[int]) -> None:
        """
        Download specific SLR datasets by ID. Requires beautifulsoup4 to parse index page,
        otherwise user must supply exact URL(s).
        """
        if not _BS4_AVAILABLE:
            self.log("⚠️  beautifulsoup4 not installed. Install it or manually download SLR zips/tars.")
            return

        base = "https://www.openslr.org"
        for slr in slr_ids:
            url = f"{base}/{slr}/"
            self.log(f"Index: {url}")
            try:
                html = self.session.get(url, timeout=60).text
                soup = BeautifulSoup(html, "html.parser")
                # naive link scrape: pick the first tar/zip
                link = soup.find("a", href=lambda h: h and (h.endswith(".zip") or h.endswith(".tar.gz") or h.endswith(".tgz")))
                if not link:
                    self.log(f"❌ No archive link found on {url}")
                    continue
                href = link.get("href")
                if href.startswith("/"):
                    href = base + href
                filename = href.split("/")[-1]
                dest = self.output_dir / "openslr" / f"SLR{slr}" / filename
                ok = self._download_stream(href, dest, desc=f"SLR{slr}")
                if ok:
                    self._extract(dest, dest.parent / "extracted")
                    self.log(f"✓ SLR{slr} extracted to {dest.parent/'extracted'}")
            except Exception as e:
                self.log(f"❌ SLR{slr} failed: {e}")

    # --------------------------
    # HF generic datasets
    # --------------------------
    def fetch_hf_dataset(self, repo_id: str, subset: Optional[str], split: List[str], out_name: Optional[str] = None) -> None:
        """
        Generic HF loader for speech datasets that expose (audio, text).
        """
        if not _HF_AVAILABLE:
            self.log("⚠️  'datasets' not installed. pip install datasets")
            return

        out_name = out_name or repo_id.replace("/", "__")
        for sp in split:
            try:
                ds = load_dataset(repo_id, subset, split=sp) if subset else load_dataset(repo_id, split=sp)
            except Exception as e:
                self.log(f"❌ HF load failed: {repo_id} ({subset}) {sp} -> {e}")
                continue

            out_dir = self.output_dir / "hf" / out_name / (subset or "default") / sp
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = []
            for i in range(len(ds)):
                row = ds[i]
                path = row.get("path") or row.get("audio", {}).get("path")
                text = row.get("sentence") or row.get("text")
                manifest.append({"path": str(path), "text": text})
            with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            self.log(f"✓ Saved manifest: {out_dir/'manifest.json'} (N={len(manifest)})")

    # --------------------------
    # Verification
    # --------------------------
    def verify_transcripts(self, manifest_path: Path, min_fraction_with_text: float = 0.8) -> Dict[str, float]:
        """
        Quick manifest sanity check: how many entries have non-empty text and existing files.
        Returns dict with counts and fractions.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        n = len(items)
        exist_cnt = 0
        text_cnt = 0
        for it in items:
            p = Path(it.get("path", ""))
            t = (it.get("text") or "").strip()
            if p.exists():
                exist_cnt += 1
            if t:
                text_cnt += 1
        frac_exist = exist_cnt / n if n else 0.0
        frac_text = text_cnt / n if n else 0.0
        if frac_text < min_fraction_with_text:
            self.log(f"⚠️  Only {frac_text:.1%} records have text; downstream WER will be impossible on missing references.")
        return {
            "n": float(n),
            "exists_fraction": float(frac_exist),
            "has_text_fraction": float(frac_text),
        }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="./datasets")
    ap.add_argument("--languages", type=str, default="hi,en", help="CSV of language codes (e.g., hi,ta,te)")
    ap.add_argument("--fetch", type=str, default="common_voice", help="CSV among: common_voice,openslr,hf")
    ap.add_argument("--slr_ids", type=str, default="", help="CSV SLR ids (e.g., 63,35)")
    ap.add_argument("--split", type=str, default="train,validation,test")
    ap.add_argument("--hf_repo", type=str, default="", help="HF repo_id (e.g., ai4bharat/indicvoxpopuli)")
    ap.add_argument("--hf_subset", type=str, default="", help="subset config for HF dataset")
    return ap.parse_args()


def main():
    args = parse_args()
    dl = DatasetDownloader(output_dir=args.output_dir, verbose=True)
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    fetches = {x.strip().lower() for x in args.fetch.split(",") if x.strip()}
    split = [x.strip() for x in args.split.split(",") if x.strip()]
    slr_ids = [int(x) for x in args.slr_ids.split(",") if x.strip().isdigit()]

    if "common_voice" in fetches:
        dl.fetch_common_voice(langs, split)

    if "openslr" in fetches and slr_ids:
        dl.fetch_openslr(slr_ids)

    if "hf" in fetches and args.hf_repo:
        subset = args.hf_subset or None
        dl.fetch_hf_dataset(args.hf_repo, subset, split)

    print("Done.")

if __name__ == "__main__":
    main()
