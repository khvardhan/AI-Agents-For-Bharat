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

# Load .env file if available
try:
    from dotenv import load_dotenv, find_dotenv
    # Try to find .env file automatically
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        _DOTENV_AVAILABLE = True
    else:
        # Try loading from current directory anyway
        load_dotenv(override=True)
        _DOTENV_AVAILABLE = True
except Exception as e:
    _DOTENV_AVAILABLE = False

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
        self.verbose = verbose
        try:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating output directory '{output_dir}': {e}")
            print(f"   Make sure you have write permissions in this location.")
            raise
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
                    for chunk in r.iter_content(chunk_size=1024 * 256):  # FIXED: was itercontent
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
        NOTE: Requires `datasets` package, internet, and HuggingFace authentication.
        """
        if not _HF_AVAILABLE:
            self.log("⚠️  'datasets' not installed. pip install datasets")
            return
        
        # Check for HF token (try multiple variable names)
        import os
        hf_token = (
            os.environ.get("HF_TOKEN") or 
            os.environ.get("HUGGING_FACE_HUB_TOKEN") or
            os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        if not hf_token:
            # Try loading .env one more time explicitly
            if _DOTENV_AVAILABLE:
                from dotenv import load_dotenv
                from pathlib import Path
                env_file = Path(".env")
                if env_file.exists():
                    self.log(f"Found .env file at: {env_file.absolute()}")
                    load_dotenv(env_file, override=True)
                    hf_token = (
                        os.environ.get("HF_TOKEN") or 
                        os.environ.get("HUGGING_FACE_HUB_TOKEN") or
                        os.environ.get("HUGGINGFACE_TOKEN")
                    )
                    if hf_token:
                        self.log("✓ Successfully loaded token from .env file")
        
        if not hf_token:
            self.log("=" * 60)
            self.log("⚠️  WARNING: No HuggingFace token found!")
            self.log("=" * 60)
            self.log("   Common Voice requires authentication. Please:")
            self.log("   1. Create account at https://huggingface.co")
            self.log("   2. Accept Common Voice terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
            self.log("   3. Get token from https://huggingface.co/settings/tokens")
            self.log("")
            self.log("   Then add token using ONE of these methods:")
            self.log("   ")
            self.log("   METHOD 1: .env file (recommended)")
            self.log("     Create a file named '.env' with:")
            self.log("     HF_TOKEN=hf_your_token_here")
            if not _DOTENV_AVAILABLE:
                self.log("     📦 First install: pip install python-dotenv")
            self.log("   ")
            self.log("   METHOD 2: Environment variable")
            self.log("     export HF_TOKEN=hf_your_token_here")
            self.log("   ")
            self.log("   METHOD 3: HuggingFace CLI")
            self.log("     huggingface-cli login")
            self.log("")
            self.log("   💡 Run check_token.py to diagnose token issues")
            self.log("=" * 60)
            self.log("")
        else:
            token_preview = hf_token[:10] + "..." if len(hf_token) > 10 else hf_token
            self.log(f"✓ HuggingFace token found: {token_preview}")
        
        for lang in languages:
            for sp in split:
                self.log(f"Fetching Common Voice: lang={lang}, split={sp}")
                ds = None
                
                # Try multiple versions with proper error messages
                versions_to_try = [
                    ("mozilla-foundation/common_voice_17_0", "17.0"),
                    ("mozilla-foundation/common_voice_16_1", "16.1"),
                    ("mozilla-foundation/common_voice_16_0", "16.0"),
                    ("mozilla-foundation/common_voice_15_0", "15.0"),
                ]
                
                for repo, ver_name in versions_to_try:
                    try:
                        ds = load_dataset(repo, lang, split=sp, token=hf_token)
                        self.log(f"✓ Loaded Common Voice {ver_name}")
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "authentication" in error_msg or "401" in error_msg or "403" in error_msg:
                            self.log(f"⚠️  Authentication required for Common Voice {ver_name}")
                            if ver_name == "17.0":  # Only show detailed help once
                                self.log("   Please authenticate with HuggingFace (see instructions above)")
                            break  # Don't try other versions if auth fails
                        elif "not found" in error_msg or "404" in error_msg:
                            self.log(f"⚠️  Language '{lang}' not available in Common Voice {ver_name}")
                        else:
                            self.log(f"⚠️  Failed to load {ver_name}: {e}")
                        continue
                    
                if ds is None:
                    self.log(f"❌ Could not load Common Voice for {lang}/{sp}")
                    self.log(f"   Try alternative datasets or check language code (should be like 'hi', 'en', 'ta', etc.)")
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

        # Get HF token if available (try multiple variable names)
        import os
        hf_token = (
            os.environ.get("HF_TOKEN") or 
            os.environ.get("HUGGING_FACE_HUB_TOKEN") or
            os.environ.get("HUGGINGFACE_TOKEN")
        )

        out_name = out_name or repo_id.replace("/", "__")
        for sp in split:
            try:
                if subset:
                    ds = load_dataset(repo_id, subset, split=sp, token=hf_token)
                else:
                    ds = load_dataset(repo_id, split=sp, token=hf_token)
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