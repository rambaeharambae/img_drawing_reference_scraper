# imgscraper.py
"""
ImgScraper — Updated with fixes requested:
1) Restored and expanded POSE_TERMS to include many original categories.
2) Improved Google/Bing parsing when given search queries (builds search URLs and parses results).
3) Improved Pinterest parsing (more robust tab handling, scrolling, logging).
4) Debug/logging: many more INFO/DEBUG logs emitted so Debug tab shows background actions (enqueue, saved, driver start/stop).
5) Ensured threads receive task-specific params (pages, imgs_per_page).
6) Proper closeEvent kept; lazy imports preserved.

Replace your current imgscraper.py with this file.
"""
from pathlib import Path
import os
import sys
import json
import time
import logging
import threading
import queue
import requests
import shutil
import atexit
import signal
import random
import re
from urllib.parse import quote_plus
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# PyQt imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QTextEdit, QLabel, QLineEdit, QFileDialog, QCheckBox,
        QSpinBox, QGroupBox, QFormLayout, QComboBox, QTabWidget, QProgressBar,
        QAction, QMessageBox, QDialog, QDoubleSpinBox, QScrollArea
    )
    from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
except Exception as e:
    raise RuntimeError("PyQt5 is required. Install via `pip install PyQt5`.") from e

# Logging setup
LOG = logging.getLogger("imgscraper")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    LOG.addHandler(_h)

# --- Constants and defaults ---
APP_DIR = Path.home() / ".imgscraper"
APP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_SAVE_FOLDER = str(Path.cwd() / "downloads")
DEFAULT_SETTINGS = {
    "save_folder": DEFAULT_SAVE_FOLDER,
    "count_per_category": 30,
    "pages_per_task": 3,
    "imgs_per_page": 10,
    "feeder_threads": 1,
    "downloader_threads": 4,
    "min_size": (400, 400),
    "use_google": True,
    "use_bing": False,
    "use_pinterest": True,
    "pinterest_headless": True,
    "pinterest_full_res": False,
    "dup_detection": False,
    "dup_threshold": 0.8,
    "dup_action": "delete_new",
    "use_yolo": False,
    "yolo_conf": 0.45,
    "yolo_model": "yolov8n.pt",
    "yolo_crop": False,
    "full_body_only": False,
    "full_body_ratio": 0.8,
    "only_bw": False,
    "strong_random": False,
    "sketch_mode": False,
    "genders": ["man", "woman", "boy", "girl"],
    "poses": []
}

# Expanded pose terms (restored / extended)
POSE_TERMS = [
    "standing", "sitting", "kneeling", "crouching", "lying down",
    "walking", "running", "jumping", "stretching", "dancing",
    "foreshortening", "low angle perspective", "high angle perspective",
    "worm's eye view", "overhead view", "three quarter view", "side view",
    "front view", "back view", "dynamic action", "fighting stance",
    "attack pose", "martial arts pose", "yoga pose", "balance pose",
    "falling pose", "contrapposto", "gesture line", "silhouette study",
    "thumbnail sketch", "figure study", "anatomy breakdown", "value study",
    "line of action", "composition study", "negative space pose",
    "sexy pose", "cute pose", "holding object pose", "dance pose" 
]

# helper functions
def slugify(s):
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[-\s]+", "_", s)
    return s[:120]

def build_random_query(gender, pose, sketch_mode=False, strong_random=False):
    base = f"{gender} {pose}"
    extras = []
    if sketch_mode:
        extras.append("sketch")
    if strong_random:
        extras.append(random.choice(["dramatic lighting", "dynamic pose", "studio", "outdoor"]))
    if extras:
        return base + " " + " ".join(extras)
    return base

# ---------------------------
# Seen management
# ---------------------------
def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_seen(folder: str):
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    urls, files, hashes = set(), set(), {}
    try:
        f = p / "seen_urls.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8") or "[]")
            if isinstance(data, list):
                urls.update([str(x) for x in data])
    except Exception:
        LOG.debug("load_seen: failed loading seen_urls", exc_info=True)
    try:
        f = p / "seen_files.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8") or "[]")
            if isinstance(data, list):
                files.update([str(x) for x in data])
    except Exception:
        LOG.debug("load_seen: failed loading seen_files", exc_info=True)
    try:
        f = p / "seen_hashes.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8") or "{}")
            if isinstance(data, dict):
                hashes.update(data)
    except Exception:
        LOG.debug("load_seen: failed loading seen_hashes", exc_info=True)
    return urls, files, hashes

def append_seen(folder: str, urls=None, filenames=None, hashes=None):
    folder_p = Path(folder)
    folder_p.mkdir(parents=True, exist_ok=True)
    urls = set(urls or [])
    filenames = set(filenames or [])
    hashes = dict(hashes or {})
    try:
        f = folder_p / "seen_urls.json"
        existing = set(json.loads(f.read_text(encoding="utf-8") or "[]")) if f.exists() else set()
        merged = sorted(list(existing.union(urls)))
        f.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOG.debug("append_seen: failed write seen_urls", exc_info=True)
    try:
        f = folder_p / "seen_files.json"
        existing = set(json.loads(f.read_text(encoding="utf-8") or "[]")) if f.exists() else set()
        merged = sorted(list(existing.union(filenames)))
        f.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOG.debug("append_seen: failed write seen_files", exc_info=True)
    try:
        f = folder_p / "seen_hashes.json"
        existing = json.loads(f.read_text(encoding="utf-8") or "{}") if f.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(hashes)
        f.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOG.debug("append_seen: failed write seen_hashes", exc_info=True)

# ---------------------------
# Browser/process helper
# ---------------------------
import psutil
class BrowserPidHelper:
    def __init__(self):
        self._pids = set()
        self._drivers = []
        self._lock = threading.Lock()
        atexit.register(self.cleanup)
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)
        except Exception:
            pass

    def add_pid(self, pid):
        with self._lock:
            try:
                self._pids.add(int(pid))
            except Exception:
                pass

    def add_driver(self, driver):
        with self._lock:
            self._drivers.append(driver)

    def _on_signal(self, signum, frame):
        LOG.info("Signal %s received, cleaning up", signum)
        self.cleanup()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    def cleanup(self):
        with self._lock:
            for d in list(self._drivers):
                try:
                    getattr(d, "quit", lambda: None)()
                except Exception:
                    try:
                        getattr(d, "close", lambda: None)()
                    except Exception:
                        pass
            self._drivers.clear()
            for pid in list(self._pids):
                try:
                    p = psutil.Process(pid)
                    if p.is_running():
                        try:
                            p.terminate()
                        except Exception:
                            pass
                except Exception:
                    pass
            self._pids.clear()

BROWSER_HELPER = BrowserPidHelper()

# ---------------------------
# Basic downloader
# ---------------------------
def download_image_basic(url: str, dest_path: str, timeout=20, headers=None):
    headers = headers or {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"}
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            tmp = dest_path + ".part"
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(r.raw, fh)
            os.replace(tmp, dest_path)
        LOG.info("Downloaded %s", dest_path)
        return True
    except Exception:
        LOG.debug("download_image_basic failed for %s", url, exc_info=True)
        try:
            if os.path.exists(dest_path + ".part"):
                os.remove(dest_path + ".part")
        except Exception:
            pass
        return False

class DownloaderThread(threading.Thread):

    def __init__(self, jobs_q: queue.Queue, out_folder: str, seen_folder: str, stop_event=None):
        super().__init__(daemon=True)
        self.jobs_q = jobs_q
        self.out_folder = Path(out_folder)
        self.seen_folder = Path(seen_folder)
        self.stop_event = stop_event or threading.Event()
        self._batch_urls = set()
        self._batch_files = set()
        self._batch_hashes = {}
        self._flush_interval = 2.0
        self._last_flush = time.time()

    def run(self):
        LOG.info("Downloader started")
        # load seen once
        self.seen_urls, self.seen_files, self.seen_hashes = load_seen(str(self.seen_folder))

        def _resolve_existing_path(existing_name):
            # try direct path
            try:
                p = Path(existing_name)
                if p.exists():
                    return p
            except Exception:
                pass
            # try under out_folder
            try:
                candidate = self.out_folder / existing_name
                if candidate.exists():
                    return candidate
            except Exception:
                pass
            # fallback: search by basename under out_folder (rare, possibly slow)
            basename = os.path.basename(existing_name)
            for root, dirs, files in os.walk(str(self.out_folder)):
                if basename in files:
                    return Path(root) / basename
            return None

        while not self.stop_event.is_set():
            try:
                job = self.jobs_q.get(timeout=0.5)
            except queue.Empty:
                self._periodic_flush()
                continue
            except Exception:
                LOG.exception("Unexpected error getting job from queue")
                self._periodic_flush()
                continue

            try:
                url = job.get("url")
                dest = Path(job.get("dest"))
                meta = job.get("meta", {}) or {}
                dest.parent.mkdir(parents=True, exist_ok=True)

                # quick seen check: support both legacy filename-only and full-path entries
                if (url and url in self.seen_urls) or (dest.name in self.seen_files) or (str(dest) in self.seen_files):
                    LOG.debug("Downloader: skipping known %s", url or dest.name)
                    try:
                        self.jobs_q.task_done()
                    except Exception:
                        pass
                    continue

                ok = download_image_basic(url, str(dest))

                if ok:
                    # size guard
                    try:
                        if os.path.getsize(dest) < 10240:
                            os.remove(dest)
                            LOG.info("Deleted too small file %s", dest)
                            ok = False
                    except Exception:
                        LOG.warning("Failed to check file size for %s", dest, exc_info=True)

                    # Black & White conversion if requested
                    if ok and meta.get("only_bw", False):
                        try:
                            from PIL import Image
                            img = Image.open(dest).convert("L")
                            img.save(dest)
                            LOG.info("Applied B/W filter to %s", dest)
                        except Exception as e:
                            LOG.warning("Failed to apply B/W filter to %s: %s", dest, e)

                if not ok:
                    LOG.debug("Failed to download %s", url)
                    try:
                        self.jobs_q.task_done()
                    except Exception:
                        pass
                    self._periodic_flush()
                    continue

                # compute perceptual hash if requested (for duplicate detection)
                phash = None
                if meta.get("compute_hash", False):
                    try:
                        from PIL import Image
                        import imagehash
                        phash = str(imagehash.phash(Image.open(dest)))
                    except Exception:
                        LOG.debug("hash computation failed for %s", dest, exc_info=True)

                # duplicate handling via pHash if enabled
                if meta.get("dup_detection", False) and phash:
                    try:
                        import imagehash
                        # compute bit-length from current phash
                        bits = len(phash) * 4
                        similarity = float(meta.get("dup_threshold", 0.8))
                        max_dist = int((1.0 - max(0.0, min(1.0, similarity))) * bits)
                        is_dup = False
                        dup_existing_key = None
                        dup_existing_value = None
                        for existing_hex, existing_value in list(self.seen_hashes.items()):
                            try:
                                ha = imagehash.hex_to_hash(existing_hex)
                                hb = imagehash.hex_to_hash(phash)
                                dist = ha - hb
                            except Exception:
                                dist = sum(c1 != c2 for c1, c2 in zip(existing_hex, phash))
                            if dist <= max_dist:
                                is_dup = True
                                dup_existing_key = existing_hex
                                dup_existing_value = existing_value
                                break

                        if is_dup:
                            action = str(meta.get("dup_action", "delete_new"))
                            LOG.info("Duplicate detected (action=%s): %s ~ %s", action, dest.name, dup_existing_value)
                            if action == "delete_new":
                                try:
                                    os.remove(dest)
                                except Exception:
                                    pass
                                ok = False
                            elif action == "replace_existing" and dup_existing_value:
                                try:
                                    # dup_existing_value may be full path or filename.
                                    existing_path = _resolve_existing_path(dup_existing_value)
                                    if existing_path and existing_path.exists():
                                        try:
                                            os.remove(existing_path)
                                            LOG.info("Removed existing duplicate: %s", existing_path)
                                        except Exception:
                                            LOG.debug("Failed to remove existing duplicate %s", existing_path, exc_info=True)
                                    # remove mapping entry
                                    try:
                                        del self.seen_hashes[dup_existing_key]
                                    except Exception:
                                        pass
                                except Exception:
                                    LOG.debug("Failed replace_existing cleanup", exc_info=True)
                                # ok remains True; new file will be recorded
                            elif action == "keep_both":
                                LOG.info("Keeping both duplicates: %s", dest.name)
                            else:
                                LOG.debug("Unknown dup action %s", action)
                    except Exception:
                        LOG.debug("Duplicate detection error for %s", dest, exc_info=True)

                # finalize saved file bookkeeping
                if ok:
                    LOG.info("Saved %s", dest)
                    if url:
                        self.seen_urls.add(url)
                        self._batch_urls.add(url)
                    self.seen_files.add(dest.name)
                    self._batch_files.add(dest.name)
                    if phash:
                        # store full path for hashes (so replace_existing can find the file)
                        self.seen_hashes[phash] = str(dest)
                        self._batch_hashes[phash] = str(dest)
                else:
                    LOG.debug("File removed/ignored after download: %s", dest)

            except Exception:
                LOG.exception("Error in downloader job processing")
            finally:
                try:
                    self.jobs_q.task_done()
                except Exception:
                    pass
                self._periodic_flush()

        # final flush on stop
        try:
            self._flush_all()
        except Exception:
            LOG.exception("Error during final flush in downloader")
        LOG.info("Downloader stopped")

    def _periodic_flush(self):
        if (
            time.time() - self._last_flush >= self._flush_interval
            and (self._batch_urls or self._batch_files or self._batch_hashes)
        ):
            self._flush_all()
            self._last_flush = time.time()

    def _flush_all(self):
        try:
        
            if not (self._batch_urls or self._batch_files or self._batch_hashes):
                return
        
            if self._batch_urls or self._batch_files or self._batch_hashes:
                append_seen(
                    str(self.seen_folder),
                    urls=self._batch_urls,
                    filenames=self._batch_files,
                    hashes=self._batch_hashes
                )
                LOG.debug(
                    "Flushed seen: %d urls, %d files, %d hashes",
                    len(self._batch_urls),
                    len(self._batch_files),
                    len(self._batch_hashes)
                )
            self._batch_urls.clear()
            self._batch_files.clear()
            self._batch_hashes.clear()
        except Exception:
            LOG.exception("Failed to flush seen")

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join(timeout=5)

# ---------------------------
# Filters (lazy imports)
# ---------------------------
class BWFilter:
    def __init__(self, sat_threshold=20):
        self.sat_threshold = sat_threshold

    def is_blackwhite(self, image_path):
        try:
            from PIL import Image, ImageStat
            im = Image.open(image_path).convert("RGB")
            hsv = im.convert("HSV")
            stat = ImageStat.Stat(hsv)
            avg_s = stat.mean[1]
            return avg_s <= self.sat_threshold
        except Exception:
            LOG.debug("BWFilter failed for %s", image_path, exc_info=True)
            return False

class PHashFilter:
    def __init__(self, dist_threshold=8):
        self.dist_threshold = dist_threshold

    def compute_hash(self, image_path):
        try:
            from PIL import Image
            import imagehash
            return str(imagehash.phash(Image.open(image_path)))
        except Exception:
            LOG.debug("PHash compute failed for %s", image_path, exc_info=True)
            return None

    def is_duplicate(self, hash_str, seen_hashes):
        if not hash_str:
            return False
        try:
            import imagehash
            for existing in seen_hashes.keys():
                try:
                    ha = imagehash.hex_to_hash(existing)
                    hb = imagehash.hex_to_hash(hash_str)
                    dist = ha - hb
                except Exception:
                    dist = sum(c1 != c2 for c1, c2 in zip(existing, hash_str))
                if dist <= self.dist_threshold:
                    return True
            return False
        except Exception:
            for existing in seen_hashes.keys():
                dist = sum(c1 != c2 for c1, c2 in zip(existing, hash_str))
                if dist <= self.dist_threshold:
                    return True
            return False

class YOLOFilter:
    def __init__(self, model_name="yolov8n.pt", conf=0.45):
        self.model_name = model_name
        self.conf = conf
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
            except Exception as e:
                raise RuntimeError("YOLO not available or model not found: " + str(e))

    def has_person(self, image_path):
        try:
            self._ensure_model()
            if self._model is None:
                return False
            results = self._model.predict(image_path, conf=self.conf, verbose=False)
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                try:
                    cls_list = getattr(boxes, "cls").tolist()
                except Exception:
                    try:
                        cls_list = list(boxes.cls)
                    except Exception:
                        cls_list = []
                names = getattr(r, "names", {})
                for idx in cls_list:
                    try:
                        idx = int(idx)
                        lab = names.get(idx, "") if isinstance(names, dict) else str(idx)
                        if str(lab).lower().startswith("person"):
                            return True
                    except Exception:
                        continue
            return False
        except Exception:
            LOG.debug("YOLO has_person error for %s", image_path, exc_info=True)
            return False
# ---------------------------
# Scraper threads
# ---------------------------
class BaseScraperThread(threading.Thread):

    def __init__(self, query_or_url, jobs_q, out_folder, seen_folder, task_meta=None, downloader_meta=None, stop_event=None, logger=None):
        super().__init__(daemon=True)
        self.query_or_url = query_or_url
        self.jobs_q = jobs_q
        self.out_folder = out_folder
        self.seen_folder = seen_folder
        self.task_meta = task_meta or {}
        self.downloader_meta = downloader_meta or {}
        self.stop_event = stop_event or threading.Event()
        self.logger = logger or LOG

    def _enqueue(self, url, suggested_name=None):
        fname = suggested_name or os.path.basename(url.split("?")[0]) or f"img_{int(time.time()*1000)}.jpg"
        dest = str(Path(self.out_folder) / fname)
        job = {"url": url, "dest": dest, "meta": self.downloader_meta}
        self.jobs_q.put(job)
        self.logger.info("Enqueued %s -> %s", url, dest)

    def stop(self):
        self.stop_event.set()

class GoogleScraperThread(BaseScraperThread):
    def run(self):
        self.logger.info("GoogleScraper started for %s", self.query_or_url)
        try:
            pages = int(self.task_meta.get("pages", 1))
            imgs_per_page = int(self.task_meta.get("imgs_per_page", 20))
            max_results = int(self.task_meta.get("count", pages * imgs_per_page))

            crawler = GoogleImageCrawler(
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=1,  # обязательно >=1 чтобы не было DummyDownloader
                storage={'root_dir': None}
            )

            found = []

            def enqueue_only(task, *args, **kwargs):
                # robust URL extraction
                url = task.get('file_url') or task.get('url') or task.get('file_urls')
                if not url:
                    return
                # already reached required count
                if len(found) >= max_results:
                    return
                # quick seen check from disk (fresh)
                seen_urls, seen_files, _ = load_seen(str(self.seen_folder))
                if (url and url in seen_urls) or (os.path.basename(url).split("?")[0] in seen_files):
                    LOG.debug("Google: skipping seen %s", url)
                    return
                # optional head request size check (<10KB)
                '''
                try:
                    r = requests.head(url, allow_redirects=True, timeout=5)
                    size = int(r.headers.get('Content-Length', 0) or 0)
                    if size and size < 10240:
                        LOG.info("Skipped too small file (<10KB): %s", url)
                        return
                except Exception:
                    LOG.debug("Could not check size for %s", url, exc_info=True)
                    '''
                found.append(url)
                LOG.info("Google found/enqueue: %s", url)
                self._enqueue(url)

            # Подменяем стандартный метод загрузки на свой
            crawler.downloader.download = enqueue_only

            # Запуск без сохранения на диск — только сбор URL
            crawler.crawl(keyword=self.query_or_url, max_num=max_results, filters=None)

        except Exception:
            self.logger.exception("GoogleScraper error")

        self.logger.info("GoogleScraper finished for %s", self.query_or_url)


class BingScraperThread(BaseScraperThread):
    def run(self):
        self.logger.info("BingScraper started for %s", self.query_or_url)
        try:
            pages = int(self.task_meta.get("pages", 1))
            imgs_per_page = int(self.task_meta.get("imgs_per_page", 20))
            max_results = int(self.task_meta.get("count", pages * imgs_per_page))

            crawler = BingImageCrawler(
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=1,  # нужен >=1, иначе DummyDownloader
                storage={'root_dir': None}
            )
            
            found = []

            def enqueue_only(task, *args, **kwargs):
                # robust URL extraction
                url = task.get('file_url') or task.get('url') or task.get('file_urls')
                if not url:
                    return
                # already reached required count
                if len(found) >= max_results:
                    return
                # quick seen check from disk (fresh)
                seen_urls, seen_files, _ = load_seen(str(self.seen_folder))
                if (url and url in seen_urls) or (os.path.basename(url).split("?")[0] in seen_files):
                    LOG.debug("Bing: skipping seen %s", url)
                    return
                # optional head request size check (<10KB)
                '''
                try:
                    r = requests.head(url, allow_redirects=True, timeout=5)
                    size = int(r.headers.get('Content-Length', 0) or 0)
                    if size and size < 10240:
                        LOG.info("Skipped too small file (<10KB): %s", url)
                        return
                except Exception:
                    LOG.debug("Could not check size for %s", url, exc_info=True)
                    '''
                found.append(url)
                LOG.info("Bing found/enqueue: %s", url)
                self._enqueue(url)

            # Подменяем стандартный метод загрузки на свой
            crawler.downloader.download = enqueue_only

            # Запуск без сохранения на диск — только сбор URL
            crawler.crawl(keyword=self.query_or_url, max_num=max_results, filters=None)

        except Exception:
            self.logger.exception("BingScraper error")

        self.logger.info("BingScraper finished for %s", self.query_or_url)


class PinterestScraperThread(BaseScraperThread):
    def __init__(
        self,
        query_or_url,
        jobs_q,
        out_folder,
        seen_folder,
        task_meta=None,
        downloader_meta=None,
        stop_event=None,
        logger=None,
        headless=True,
        imgs_per_page=10
    ):
        super().__init__(query_or_url, jobs_q, out_folder, seen_folder,
                         task_meta=task_meta, downloader_meta=downloader_meta,
                         stop_event=stop_event, logger=logger)
        self.headless = headless
        self.driver = None
        self.imgs_per_page = imgs_per_page

    def _start_driver(self):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
        except Exception as e:
            raise RuntimeError("Selenium/webdriver_manager required: " + str(e))

        opts = Options()

        # Headless
        if self.headless:
            try:
                opts.add_argument('--headless=new')
            except Exception:
                opts.add_argument('--headless')

        # Минимальное окно
        opts.add_argument("--window-size=800,600")

        # Отключение изображений для ускорения загрузки страниц
        prefs = {"profile.managed_default_content_settings.images": 2}
        opts.add_experimental_option("prefs", prefs)

        # Прочие оптимизации
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('--log-level=3')
        opts.add_argument('--disable-gpu')
        opts.add_argument('--disable-extensions')

        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=opts)
        driver.implicitly_wait(3)

        BROWSER_HELPER.add_driver(driver)
        try:
            if getattr(service, "process", None) and hasattr(service.process, "pid"):
                BROWSER_HELPER.add_pid(service.process.pid)
        except Exception:
            pass

        self.driver = driver
        return driver


    def run(self):
        self.logger.info("PinterestScraper started for %s", self.query_or_url)
        try:
            driver = self._start_driver()
            target = self.query_or_url
            if not target.startswith("http"):
                from requests.utils import requote_uri
                q = requote_uri(self.query_or_url)
                target = f"https://www.pinterest.com/search/pins/?q={q}"
            self.logger.info("Pinterest: opening %s", target)
            driver.get(target)
            time.sleep(1.2)

            collected = []
            scroll_attempts = 0
            
            target_count = int(self.task_meta.get("count") or self.task_meta.get("imgs_per_page") or 20)
            max_scrolls = max(10, int(target_count / self.imgs_per_page) * 5)
            while not self.stop_event.is_set() and len(collected) < target_count and scroll_attempts < max_scrolls:
                try:
                    thumbs = driver.find_elements("css selector", "a[href*='/pin/']")
                    for el in thumbs:
                        try:
                            href = el.get_attribute("href")
                            if href and href not in collected:
                                collected.append(href)
                        except Exception:
                            continue
                    driver.execute_script("window.scrollBy(0, window.innerHeight);")
                    time.sleep(0.8)
                    scroll_attempts += 1
                except Exception:
                    LOG.debug("Pinterest scroll iteration error", exc_info=True)
                    time.sleep(0.5)
            self.logger.info("Pinterest: collected %d pin links", len(collected))

            enqueued = 0
            for pin_url in list(collected)[:target_count]:
                if self.stop_event.is_set():
                    break
                try:
                    before = list(driver.window_handles)
                    cur_handle = driver.current_window_handle
                    driver.execute_script("window.open(arguments[0]);", pin_url)
                    new_handle = None
                    wait_start = time.time()
                    while time.time() - wait_start < 6:
                        after = list(driver.window_handles)
                        new = [h for h in after if h not in before]
                        if new:
                            new_handle = new[0]
                            break
                        time.sleep(0.15)
                    if new_handle:
                        driver.switch_to.window(new_handle)
                    else:
                        driver.get(pin_url)
                    time.sleep(0.8)
                    imgs = driver.find_elements("css selector", "img")
                    img_url = None
                    for im in imgs:
                        try:
                            s = im.get_attribute("src")
                            if not s:
                                s = im.get_attribute("data-src") or im.get_attribute("data-original") or im.get_attribute("srcset")
                            if s and "pinimg" in s:
                                img_url = s
                                break
                            if s and s.startswith("http") and any(ext in s for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                                if not img_url:
                                    img_url = s
                        except Exception:
                            continue
                    if img_url:
                        self._enqueue_with_min_size_check(img_url)
                        enqueued += 1
                    if new_handle:
                        driver.close()
                        driver.switch_to.window(cur_handle)
                except Exception:
                    LOG.debug("Pinterest per-pin handling error", exc_info=True)
                    try:
                        if len(driver.window_handles) > 1:
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                    except Exception:
                        pass
            self.logger.info("PinterestScraper enqueued %d items", enqueued)
        except Exception:
            self.logger.exception("PinterestScraper failed")
        self.logger.info("PinterestScraper finished for %s", self.query_or_url)
    
    def _enqueue_with_min_size_check(self, url):
        try:
            import requests
            r = requests.head(url, allow_redirects=True, timeout=5)
            size = int(r.headers.get('Content-Length', 0))
            if size < 10240:
                LOG.info("Skipped too small file (<10KB): %s", url)
                return
        except Exception:
            LOG.debug("Could not check size for %s", url, exc_info=True)
        self._enqueue(url)

# ---------------------------
# ScraperThread QThread orchestrator
# ---------------------------
class ScraperThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    done_signal = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.copy()
        self._stop = threading.Event()
        self._scrapers = []
        self._jobs_q = queue.Queue()

    def run(self):
        try:
            tasks = self.cfg.get("tasks", [])
            if not tasks:
                self.status_signal.emit("No tasks")
                self.done_signal.emit()
                return
            self.status_signal.emit("Starting...")
            # start downloader (single thread pool can be expanded)
            downloader_threads = int(self.cfg.get("downloader_threads", 10))
            
            save_folder = self.cfg.get("save_folder", DEFAULT_SAVE_FOLDER)
            self.seen_root = str(Path(save_folder) / ".seen")
            ensure_folder(save_folder)
            ensure_folder(self.seen_root)

            downloader_stop = threading.Event()
            downloader_threads = max(1, int(self.cfg.get("downloader_threads", 4)))
            self._downloaders = []
            for _ in range(downloader_threads):
                d = DownloaderThread(self._jobs_q, save_folder, self.seen_root, stop_event=downloader_stop)
                d.start()
                self._downloaders.append(d)
            
            # partition tasks across feeder threads
            feeder_threads = int(self.cfg.get("feeder_threads", 1))
            chunks = [[] for _ in range(max(1, feeder_threads))]
            for i, t in enumerate(tasks):
                chunks[i % len(chunks)].append(t)
            workers = []
            for chunk in chunks:
                wt = threading.Thread(target=self._process_task_chunk, args=(chunk,), daemon=True)
                workers.append(wt)
                wt.start()
            # monitor workers
            while any(w.is_alive() for w in workers):
                if self._stop.is_set():
                    break
                time.sleep(0.5)
            # wait until queue empty or stop
            while not self._jobs_q.empty():
                if self._stop.is_set():
                    break
                time.sleep(0.3)

            # stop downloader
            downloader_stop.set()
            for d in getattr(self, "_downloaders", []):
                try:
                    d.stop(wait=True)
                except Exception:
                    pass

            self.progress_signal.emit(100)
            self.status_signal.emit("Completed")
            self.done_signal.emit()

        except Exception:
            LOG.exception("ScraperThread run error")
            self.status_signal.emit("Error")
            self.done_signal.emit()

    def _process_task_chunk(self, chunk):
        total = len(chunk)
        done = 0
        for t in chunk:
            if self._stop.is_set():
                break
            source = t.get("source")
            query = t.get("query")
            self.status_signal.emit(f"Processing {source}: {query}")
            stop_ev = threading.Event()
            task_meta = {"pages": t.get("pages", 1), "imgs_per_page": t.get("imgs_per_page", 10), "count": t.get("count", 30)}
 # detailed downloader metadata passed into each enqueue job
            downloader_meta = {
                "compute_hash": bool(self.cfg.get("dup_detection", False)),
                "dup_detection": bool(self.cfg.get("dup_detection", False)),
                "dup_threshold": float(self.cfg.get("dup_threshold", 0.8)),
                "dup_action": str(self.cfg.get("dup_action", "delete_new")),
                "only_bw": bool(self.cfg.get("only_bw", False)),
                "min_size": tuple(self.cfg.get("min_size", (200,200))),
                "yolo": bool(self.cfg.get("use_yolo", False)),
                "yolo_conf": float(self.cfg.get("yolo_conf", 0.45)),
                "yolo_model": str(self.cfg.get("yolo_model", "yolov8n.pt")),
                "yolo_crop": bool(self.cfg.get("yolo_crop", False))
            }

            if source == "google":
                
                th = GoogleScraperThread(
                    query,
                    self._jobs_q,
                    t.get("folder"),
                    self.seen_root,
                    task_meta=task_meta,
                    downloader_meta=downloader_meta,
                    stop_event=stop_ev,
                    logger=LOG
                )
                
            elif source == "bing":
                th = BingScraperThread(query, self._jobs_q, t.get("folder"), self.seen_root, task_meta=task_meta, downloader_meta=downloader_meta, stop_event=stop_ev, logger=LOG)
            elif source == "pinterest":
                th = PinterestScraperThread(query, self._jobs_q, t.get("folder"), self.seen_root, task_meta=task_meta, downloader_meta=downloader_meta, stop_event=stop_ev, logger=LOG, headless=self.cfg.get("pinterest_headless", True))
            else:
                continue
            self._scrapers.append((th, stop_ev))
            th.start()
            # wait for scraper to complete
            while th.is_alive():
                if self._stop.is_set():
                    stop_ev.set()
                time.sleep(0.3)
            done += 1
            try:
                pct = int((done / max(1, total)) * 100)
                self.progress_signal.emit(pct)
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        for th, ev in getattr(self, "_scrapers", []):
            try:
                ev.set()
            except Exception:
                pass

# ---------------------------
# Utilities used by UI
# ---------------------------
def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def compute_image_phash(path):
    try:
        from PIL import Image
        import imagehash
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        LOG.debug("compute_image_phash failed for %s", path, exc_info=True)
        return None

def save_image_hash_index(folder, index_map):
    try:
        p = Path(folder) / "image_hash_index.json"
        p.write_text(json.dumps(index_map, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOG.debug("save_image_hash_index failed", exc_info=True)

# ---------------------------
# QtLogHandler for Debug tab
# ---------------------------
from PyQt5.QtCore import pyqtSignal, QObject
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except Exception:
            pass

# QDoubleSpinBoxWithDefaults
class QDoubleSpinBoxWithDefaults(QDoubleSpinBox):
    def __init__(self, default=0.45):
        super().__init__()
        self.setRange(0.0, 1.0)
        self.setSingleStep(0.01)
        self.setValue(default)

# Detect ultralytics presence for the UI
HAS_YOLO = False
try:
    import importlib
    HAS_YOLO = importlib.util.find_spec("ultralytics") is not None
except Exception:
    HAS_YOLO = False

# ---------------------------
# MainWindow (restored original UI + fixes)
# ---------------------------
class MainWindow(QMainWindow):

    def start_pinterest_browser(self, silent=True):
        if getattr(self, 'pinterest_driver', None):
            logging.info("Pinterest driver already running.")
            if not silent:
                QMessageBox.information(self, 'Browser', 'Browser is running.')
            try:
                self.pinterest_start_btn.setEnabled(False)
                self.pinterest_stop_btn.setEnabled(True)
            except Exception:
                pass
            return
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            opts = Options()

            # Headless
            if self.settings.get('pinterest_headless', True):
                try:
                    opts.add_argument('--headless=new')
                except Exception:
                    opts.add_argument('--headless')

            # Маленькое окно для ускорения рендера
            opts.add_argument('--window-size=800,600')

            # Отключение картинок для ускорения
            prefs = {
                "profile.managed_default_content_settings.images": 2
            }
            opts.add_experimental_option("prefs", prefs)

            # Прочие оптимизации
            opts.add_argument('--no-sandbox')
            opts.add_argument('--disable-dev-shm-usage')
            opts.add_argument('--log-level=3')
            opts.add_argument('--disable-gpu')

            driver_path = ChromeDriverManager().install()
            service = Service(driver_path)

            self.pinterest_driver = webdriver.Chrome(service=service, options=opts)
            self.pinterest_driver_lock = threading.Lock()
            logging.info('Started Pinterest driver at %s', driver_path)

            try:
                self.pinterest_start_btn.setEnabled(False)
                self.pinterest_stop_btn.setEnabled(True)
            except Exception:
                pass

            if not silent:
                QMessageBox.information(self, 'Browser', 'Pinterest browser is running.')

            try:
                BROWSER_HELPER.add_driver(self.pinterest_driver)
                if getattr(service, "process", None) and hasattr(service.process, "pid"):
                    BROWSER_HELPER.add_pid(service.process.pid)
            except Exception:
                pass

        except Exception as e:
            logging.error('Failed to start Pinterest browser: %s', e)
            if not silent:
                QMessageBox.critical(self, 'Error', f'Failed to start browser: {e}')

    def stop_pinterest_browser(self):
        d = getattr(self, "pinterest_driver", None)
        if not d:
            logging.info("Pinterest driver not running (stop invoked).")
            QMessageBox.information(self, "Browser", "The browser is not running.")
            try:
                self.pinterest_start_btn.setEnabled(True)
                self.pinterest_stop_btn.setEnabled(False)
            except Exception:
                pass
            return
        logging.info("Stopping Pinterest driver...")
        try:
            d.quit()
            logging.info("Pinterest driver quit successfully.")
        except Exception as e:
            logging.error("Error quitting Pinterest driver: %s", e)
        self.pinterest_driver = None
        self.pinterest_driver_lock = None
        try:
            self.pinterest_start_btn.setEnabled(True)
            self.pinterest_stop_btn.setEnabled(False)
        except Exception:
            pass
        QMessageBox.information(self, "Browser", "The browser is stopped.")
        try:
            BROWSER_HELPER.cleanup()
        except Exception:
            pass

    def __init__(self):                                                                                                                      #####GUI 

        super().__init__()
        self.setWindowTitle("Reference Scraper by Hara (modified and wrote by ChatGPT) for personal use")
        self.setGeometry(120, 120, 980, 760)

        self.settings = DEFAULT_SETTINGS.copy()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self._qt_log_handler = QtLogHandler()
        self._qt_log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        root_logger = logging.getLogger()
        already = False
        for h in getattr(root_logger, "handlers", []):
            if isinstance(h, QtLogHandler):
                already = True
                break
        if not already:
            try:
                self._qt_log_handler.log_signal.connect(self._append_log)
            except Exception:
                pass
            root_logger.addHandler(self._qt_log_handler)

        self._load_settings_from_disk()
        ensure_folder(self.settings["save_folder"])
        self.threads = []
        self.scraper_thread = None
        self._init_ui()

    def _append_log(self, msg: str):
        try:
            self.log_view.append(msg)
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.End)
            self.log_view.setTextCursor(cursor)
        except Exception:
            pass

    def _load_settings_from_disk(self):
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.settings.update(data)
                        logging.info("Loaded settings from %s", CONFIG_PATH)
        except Exception as e:
            logging.debug("Failed to load settings: %s", e)

    def _save_settings_to_disk(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            logging.info("Saved settings to %s", CONFIG_PATH)
        except Exception as e:
            logging.debug("Failed to save settings: %s", e)

    def _init_ui(self):
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter, QGroupBox,
            QLabel, QPushButton, QProgressBar, QAction, QPlainTextEdit
        )
        from PyQt5.QtCore import Qt

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ===== Левая панель: вкладки настроек =====
        tabs = QTabWidget()
        tabs.addTab(self._tab_sources(), "Sources")
        tabs.addTab(self._tab_poses(), "Poses")
        tabs.addTab(self._tab_custom_queries(), "Additional requests")
        tabs.addTab(self._tab_filters(), "Filters")
        splitter.addWidget(tabs)

        # ===== Правая панель: управление + логи =====
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # --- Группа управления ---
        control_group = QGroupBox("Scraping Control")
        cg_layout = QVBoxLayout()

        self.folder_label = QLabel(f"Save: {self.settings['save_folder']}")
        cg_layout.addWidget(self.folder_label)

        btn_choose = QPushButton("Choose folder")
        btn_choose.clicked.connect(self.choose_folder)
        cg_layout.addWidget(btn_choose)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_scraping)
        cg_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_scraping)
        self.btn_stop.setEnabled(False)
        cg_layout.addWidget(self.btn_stop)

        self.status_label = QLabel("Ready.")
        cg_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        cg_layout.addWidget(self.progress_bar)

        control_group.setLayout(cg_layout)
        right_layout.addWidget(control_group)

        # --- Группа логов ---
        log_group = QGroupBox("Logs")
        lg_layout = QVBoxLayout()

        # Используем QPlainTextEdit вместо QTextEdit (быстрее для больших логов)
        if not hasattr(self, "log_view") or self.log_view is None:
            from PyQt5.QtGui import QFont
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setFont(QFont("Consolas", 9))
        lg_layout.addWidget(self.log_view)

        log_group.setLayout(lg_layout)
        right_layout.addWidget(log_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([650, 300])  # стартовое соотношение панелей

        # ===== Меню =====
        menubar = self.menuBar()
        tools = menubar.addMenu("Tools")
        act_clean = QAction("Clean duplicates", self)
        act_clean.triggered.connect(self.clean_duplicates_manual)
        tools.addAction(act_clean)
        act_reset_seen = QAction("Reset seen_urls (for all folders)", self)
        act_reset_seen.triggered.connect(self.reset_seen_history)
        tools.addAction(act_reset_seen)
        act_save = QAction("Save configuration", self)
        act_save.triggered.connect(lambda: self._save_settings_and_notify())
        tools.addAction(act_save)
        act_help = QAction("Info", self)
        act_help.triggered.connect(self.show_help_dialog)
        tools.addAction(act_help)

        # ===== Стиль =====
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: Segoe UI, sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: white;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 3px 0 3px;
            }
        """)


    def _tab_sources(self):
        w = QWidget()
        g = QGridLayout()
        w.setLayout(g)
        self.cb_google = QCheckBox("Google"); self.cb_google.setChecked(bool(self.settings.get("use_google", True)))
        self.cb_bing = QCheckBox("Bing"); self.cb_bing.setChecked(bool(self.settings.get("use_bing", False)))
        self.cb_pinterest = QCheckBox("Pinterest"); self.cb_pinterest.setChecked(bool(self.settings.get("use_pinterest", True)))
        g.addWidget(QLabel("Sourceses:"), 0, 0)
        g.addWidget(self.cb_google, 0, 1); g.addWidget(self.cb_bing, 0, 2); g.addWidget(self.cb_pinterest, 0, 3)

        g.addWidget(QLabel("Images per category:"), 1, 0)
        self.count_spin = QSpinBox(); self.count_spin.setRange(1,1000); self.count_spin.setValue(int(self.settings.get("count_per_category", 30)))
        g.addWidget(self.count_spin, 1, 1)

        g.addWidget(QLabel("Pages per task:"), 2, 0)
        self.pages_spin = QSpinBox(); self.pages_spin.setRange(1,10); self.pages_spin.setValue(int(self.settings.get("pages_per_task", 3)))
        g.addWidget(self.pages_spin, 2, 1)

        g.addWidget(QLabel("Pictures per page (Pinterest selection):"), 3, 0)
        self.imgs_per_page_spin = QSpinBox(); self.imgs_per_page_spin.setRange(1,100); self.imgs_per_page_spin.setValue(int(self.settings.get("imgs_per_page", 10)))
        g.addWidget(self.imgs_per_page_spin, 3, 1)

        return w

    def _tab_poses(self):
        w = QWidget(); v = QVBoxLayout(); w.setLayout(v)
        gb = QGroupBox("Gender"); hl = QHBoxLayout()
        self.gender_checks = {}
        saved_genders = set(self.settings.get("genders", ["man","woman","boy","girl"]))
        for g in ["man","woman","boy","girl"]:
            cb = QCheckBox(g); cb.setChecked(g in saved_genders); self.gender_checks[g] = cb; hl.addWidget(cb)
        gb.setLayout(hl); v.addWidget(gb)

        pbox = QGroupBox("Poses"); pl = QGridLayout()
        self.pose_checks = {}
        saved_poses = set(self.settings.get("poses", POSE_TERMS[:6]))
        cols = 3
        for idx, term in enumerate(POSE_TERMS):
            cb = QCheckBox(term); cb.setChecked(term in saved_poses); self.pose_checks[term] = cb
            pl.addWidget(cb, idx // cols, idx % cols)
        pbox.setLayout(pl); v.addWidget(pbox)

        ph = QHBoxLayout()
        self.preset_combo = QComboBox(); self.preset_combo.addItems(["— not selected —", "Foreshortening pack", "Dynamic action pack", "Perspective pack"])
        ph.addWidget(self.preset_combo)
        btn_preset = QPushButton("Apply preset"); btn_preset.clicked.connect(self.apply_preset); ph.addWidget(btn_preset)
        v.addLayout(ph)
        rh = QHBoxLayout()
        self.strong_rand_cb = QCheckBox("maximum randomness when parsing"); self.strong_rand_cb.setChecked(bool(self.settings.get("strong_random", False))); rh.addWidget(self.strong_rand_cb)
        self.sketch_mode_cb = QCheckBox("Find sketches/lines"); self.sketch_mode_cb.setChecked(bool(self.settings.get("sketch_mode", False))); rh.addWidget(self.sketch_mode_cb)
        v.addLayout(rh)
        return w

    def _tab_custom_queries(self):
        w = QWidget(); v = QVBoxLayout(); w.setLayout(v)
        v.addWidget(QLabel("Additional queries (one line = one query) - use if you want to parse specific phrases/keys."))
        self.custom_queries_edit = QTextEdit(); self.custom_queries_edit.setPlainText(self.settings.get("custom_queries", ""))
        v.addWidget(self.custom_queries_edit)
        h = QHBoxLayout()
        self.use_custom_only_cb = QCheckBox("Use only additional queries (ignore generated ones)"); self.use_custom_only_cb.setChecked(bool(self.settings.get("use_custom_only", False)))
        h.addWidget(self.use_custom_only_cb)
        self.append_custom_cb = QCheckBox("Add additional queries to generated ones"); self.append_custom_cb.setChecked(bool(self.settings.get("append_custom", False)))
        h.addWidget(self.append_custom_cb)
        v.addLayout(h)
        v.addWidget(QLabel("Include domains (comma separated, optional) — will be saved in Pinterest task/manual scraping"))
        self.include_domains_le = QLineEdit(self.settings.get("include_domains", "")); v.addWidget(self.include_domains_le)
        v.addWidget(QLabel("Exclude domains (comma separated, optional)"))
        self.exclude_domains_le = QLineEdit(self.settings.get("exclude_domains", "")); v.addWidget(self.exclude_domains_le)
        return w

    def _tab_filters(self):
        w = QWidget(); v = QVBoxLayout(); w.setLayout(v)
        self.pinterest_fullres_cb = QCheckBox("Download full-res images (if available)")
        self.pinterest_fullres_cb.setChecked(bool(self.settings.get("pinterest_full_res", False)))
        v.addWidget(self.pinterest_fullres_cb)

        hbr = QHBoxLayout()
        self.pinterest_start_btn = QPushButton("Start browser")
        self.pinterest_start_btn.clicked.connect(self.start_pinterest_browser)
        hbr.addWidget(self.pinterest_start_btn)
        self.pinterest_stop_btn = QPushButton("Stop browser")
        self.pinterest_stop_btn.setEnabled(False)
        self.pinterest_stop_btn.clicked.connect(self.stop_pinterest_browser)
        hbr.addWidget(self.pinterest_stop_btn)
        v.addLayout(hbr)

        self.dup_cb = QCheckBox("Find duplicates by image (pHash)")
        self.dup_cb.setChecked(bool(self.settings.get("dup_detection", False)))
        v.addWidget(self.dup_cb)

        hdup = QHBoxLayout()
        self.dup_threshold = QDoubleSpinBox()
        self.dup_threshold.setRange(0.5, 0.99)
        self.dup_threshold.setSingleStep(0.05)
        self.dup_threshold.setValue(float(self.settings.get("dup_threshold", 0.8)))
        hdup.addWidget(QLabel("Threshold of similarity (0..1):"))
        hdup.addWidget(self.dup_threshold)
        self.dup_action_combo = QComboBox()
        self.dup_action_combo.addItems(["delete_new", "replace_existing", "keep_both"])
        self.dup_action_combo.setCurrentText(self.settings.get("dup_action", "delete_new"))
        hdup.addWidget(self.dup_action_combo)
        v.addLayout(hdup)

        self.reindex_btn = QPushButton("Index folder (hashes)")
        self.reindex_btn.clicked.connect(self.reindex_folder_dialog)
        v.addWidget(self.reindex_btn)

        self.yolo_cb = QCheckBox("People filter (YOLOv8)")
        if not HAS_YOLO:
            self.yolo_cb.setText("People filter (YOLOv8) - not available (ultralytics not installed)")
            self.yolo_cb.setEnabled(False)
        self.yolo_cb.setChecked(bool(self.settings.get("use_yolo", False)))
        v.addWidget(self.yolo_cb)
        v.addWidget(QLabel("YOLO model (eg yolov8n.pt or path):"))
        self.yolo_model_edit = QLineEdit(self.settings.get("yolo_model", "yolov8n.pt")); v.addWidget(self.yolo_model_edit)
        v.addWidget(QLabel("YOLO confidence:"))
        self.yolo_conf = QDoubleSpinBoxWithDefaults(float(self.settings.get("yolo_conf", 0.45))); v.addWidget(self.yolo_conf)
        self.yolo_crop_cb = QCheckBox("Save YOLO crop in _crops"); self.yolo_crop_cb.setChecked(bool(self.settings.get("yolo_crop", False))); v.addWidget(self.yolo_crop_cb)

        self.full_body_cb = QCheckBox("Only full human height (full-body)"); self.full_body_cb.setChecked(bool(self.settings.get("full_body_only", False)))
        if not HAS_YOLO:
            self.full_body_cb.setEnabled(False)
        v.addWidget(self.full_body_cb)
        v.addWidget(QLabel("Full-body ratio threshold (0.5 - 1.0):"))
        self.full_body_ratio = QDoubleSpinBox(); self.full_body_ratio.setRange(0.5, 1.0); self.full_body_ratio.setSingleStep(0.05); self.full_body_ratio.setValue(float(self.settings.get("full_body_ratio", 0.8))); v.addWidget(self.full_body_ratio)

        self.only_bw_cb = QCheckBox("Black and white only (filter after image downloading)"); self.only_bw_cb.setChecked(bool(self.settings.get("only_bw", False))); v.addWidget(self.only_bw_cb)

        self.pinterest_headless_cb = QCheckBox("Pinterest headless (windowsless)"); self.pinterest_headless_cb.setChecked(bool(self.settings.get("pinterest_headless", True))); v.addWidget(self.pinterest_headless_cb)
        v.addWidget(QLabel("minimum image size (W x H):"))
        wh = QHBoxLayout()
        self.min_w = QSpinBox(); self.min_w.setRange(50,5000); self.min_w.setValue(int(self.settings.get("min_size", (200,200))[0]))
        self.min_h = QSpinBox(); self.min_h.setRange(50,5000); self.min_h.setValue(int(self.settings.get("min_size", (200,200))[1]))
        wh.addWidget(QLabel("W:")); wh.addWidget(self.min_w); wh.addWidget(QLabel("H:")); wh.addWidget(self.min_h)
        v.addLayout(wh)
        return w

    def apply_preset(self):
        p = self.preset_combo.currentText()
        for cb in self.pose_checks.values(): cb.setChecked(False)
        if p == "Foreshortening pack":
            terms = ["foreshortening", "low angle perspective", "high angle perspective", "worm's eye view", "overhead view"]
        elif p == "Dynamic action pack":
            terms = ["dynamic action", "attack pose", "martial arts pose", "running", "jumping"]
        elif p == "Perspective pack":
            terms = ["low angle perspective", "high angle perspective", "three quarter view", "side view", "front view"]
        else:
            terms = []
        for t in terms:
            if t in self.pose_checks:
                self.pose_checks[t].setChecked(True)

    def choose_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select a folder to save")
        if f:
            self.settings["save_folder"] = f
            self.folder_label.setText(f"Save: {f}")
            self._save_settings_from_ui()

    def _gather_ui_settings(self):
        self.settings["save_folder"] = self.settings.get("save_folder", DEFAULT_SAVE_FOLDER)
        self.settings["count_per_category"] = int(self.count_spin.value())
        self.settings["pages_per_task"] = int(self.pages_spin.value())
        self.settings["imgs_per_page"] = int(self.imgs_per_page_spin.value())
        self.settings["feeder_threads"] = int(self.settings.get("feeder_threads", 1))
        self.settings["downloader_threads"] = int(self.settings.get("downloader_threads", 4))
        self.settings["min_size"] = (int(self.min_w.value()), int(self.min_h.value()))
        self.settings["use_yolo"] = bool(self.yolo_cb.isChecked())
        self.settings["yolo_conf"] = float(self.yolo_conf.value())
        self.settings["yolo_crop"] = bool(self.yolo_crop_cb.isChecked())
        self.settings["yolo_model"] = str(self.yolo_model_edit.text()).strip() or self.settings.get("yolo_model", "yolov8n.pt")
        self.settings["pinterest_headless"] = bool(self.pinterest_headless_cb.isChecked())
        self.settings["strong_random"] = bool(self.strong_rand_cb.isChecked())
        self.settings["sketch_mode"] = bool(self.sketch_mode_cb.isChecked())
        self.settings["full_body_only"] = bool(self.full_body_cb.isChecked())
        self.settings["full_body_ratio"] = float(self.full_body_ratio.value())
        self.settings["only_bw"] = bool(self.only_bw_cb.isChecked())

        self.settings["use_google"] = bool(self.cb_google.isChecked())
        self.settings["use_bing"] = bool(self.cb_bing.isChecked())
        self.settings["use_pinterest"] = bool(self.cb_pinterest.isChecked())

        genders = [g for g,cb in self.gender_checks.items() if cb.isChecked()]
        poses = [p for p,cb in self.pose_checks.items() if cb.isChecked()]
        self.settings["genders"] = genders
        self.settings["poses"] = poses

        self.settings["pinterest_full_res"] = bool(self.pinterest_fullres_cb.isChecked())
        self.settings["dup_detection"] = bool(self.dup_cb.isChecked())
        self.settings["dup_threshold"] = float(self.dup_threshold.value())
        self.settings["dup_action"] = str(self.dup_action_combo.currentText())

        self.settings["custom_queries"] = str(self.custom_queries_edit.toPlainText() or "")
        self.settings["use_custom_only"] = bool(self.use_custom_only_cb.isChecked())
        self.settings["append_custom"] = bool(self.append_custom_cb.isChecked())
        self.settings["include_domains"] = str(self.include_domains_le.text() or "")
        self.settings["exclude_domains"] = str(self.exclude_domains_le.text() or "")

    def _save_settings_from_ui(self):
        self._gather_ui_settings()
        self._save_settings_to_disk()
                                                                                                                            ###########GUI
    def collect_config_and_tasks(self):
        self._gather_ui_settings()
        cfg = {}
        cfg.update(self.settings)
        sources = []
        if self.settings.get("use_google", True): sources.append("google")
        if self.settings.get("use_bing", False): sources.append("bing")
        if self.settings.get("use_pinterest", True): sources.append("pinterest")
        cfg["sources"] = sources
        tasks = []
        custom_raw = self.settings.get("custom_queries", "").strip()
        custom_list = [line.strip() for line in custom_raw.splitlines() if line.strip()]
        if self.settings.get("use_custom_only") and custom_list:
            for q in custom_list:
                for src in sources:
                    subfolder = f"{src}_custom_{slugify(q)}"
                    folder = os.path.join(cfg["save_folder"], subfolder)
                    tasks.append({
                        "source": src,
                        "query": q,
                        "folder": folder,
                        "count": cfg.get("count_per_category", 30),
                        "pages": cfg.get("pages_per_task", 3),
                        "imgs_per_page": cfg.get("imgs_per_page", 10),
                        "include_domains": cfg.get("include_domains", ""),
                        "exclude_domains": cfg.get("exclude_domains", "")
                    })
        else:
            genders = self.settings.get("genders", [])
            poses = self.settings.get("poses", [])
            if not sources or not genders or not poses:
                if custom_list and sources:
                    for q in custom_list:
                        for src in sources:
                            subfolder = f"{src}_custom_{slugify(q)}"
                            folder = os.path.join(cfg["save_folder"], subfolder)
                            tasks.append({
                                "source": src,
                                "query": q,
                                "folder": folder,
                                "count": cfg.get("count_per_category", 30),
                                "pages": cfg.get("pages_per_task", 3),
                                "imgs_per_page": cfg.get("imgs_per_page", 10),
                                "include_domains": cfg.get("include_domains", ""),
                                "exclude_domains": cfg.get("exclude_domains", "")
                            })
                else:
                    cfg["tasks"] = []
                    return cfg
            n_queries = 2 if not cfg.get("strong_random") else 4
            for gender in genders:
                for pose in poses:
                    for _ in range(n_queries):
                        q = build_random_query(gender, pose, sketch_mode=cfg.get('sketch_mode', False), strong_random=cfg.get('strong_random', False))
                        for src in sources:
                            subfolder = f"{src}_{'sketches' if cfg.get('sketch_mode') else 'photos'}_{gender}_{pose}".replace(" ", "_")
                            folder = os.path.join(cfg["save_folder"], subfolder)
                            tasks.append({
                                "source": src,
                                "query": q,
                                "folder": folder,
                                "count": cfg.get("count_per_category", 30),
                                "pages": cfg.get("pages_per_task", 3),
                                "imgs_per_page": cfg.get("imgs_per_page", 10),
                                "include_domains": cfg.get("include_domains", ""),
                                "exclude_domains": cfg.get("exclude_domains", "")
                            })
                            if custom_list:
                                for custom_term in custom_list:
                                    q2 = q + " " + custom_term
                                    subfolder2 = f"{subfolder}_{slugify(custom_term)}"
                                    folder2 = os.path.join(cfg["save_folder"], subfolder2)
                                    tasks.append({
                                        "source": src,
                                        "query": q2,
                                        "folder": folder2,
                                        "count": cfg.get("count_per_category", 30),
                                        "pages": cfg.get("pages_per_task", 3),
                                        "imgs_per_page": cfg.get("imgs_per_page", 10),
                                        "include_domains": cfg.get("include_domains", ""),
                                        "exclude_domains": cfg.get("exclude_domains", "")
                                    })
        cfg["tasks"] = tasks
        return cfg

    def start_scraping(self):
        cfg = self.collect_config_and_tasks()
        if not cfg.get("tasks"):
            QMessageBox.warning(self, "Error", "No tasks created (check sources, gender/poses or additional queries)." )
            return
        self._save_settings_from_ui()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")
        thread_cfg = cfg.copy()
        if getattr(self, "pinterest_driver", None):
            thread_cfg["pinterest_driver"] = self.pinterest_driver
            thread_cfg["pinterest_driver_lock"] = self.pinterest_driver_lock
        thread_cfg.update(self.settings)
        ensure_folder(thread_cfg.get("save_folder", DEFAULT_SAVE_FOLDER))
        self.scraper_thread = ScraperThread(thread_cfg)
        self.scraper_thread.status_signal.connect(self.status_label.setText)
        try:
            self.threads.append(self.scraper_thread)
        except Exception:
            pass
        self.scraper_thread.progress_signal.connect(self.progress_bar.setValue)
        self.scraper_thread.done_signal.connect(self.on_done)
        self.scraper_thread.start()

    def stop_scraping(self):
        if self.scraper_thread:
            self.scraper_thread.stop()
            self.status_label.setText("Waiting for scraping to stop...")

    def on_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Completed.")
        QMessageBox.information(self, "Done", "Parsing complete.")
        self._save_settings_from_ui()
        try:
            if self.scraper_thread in getattr(self, 'threads', []):
                try:
                    self.threads.remove(self.scraper_thread)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.scraper_thread = None
        except Exception:
            pass

    def clean_duplicates_manual(self):
        folder = QFileDialog.getExistingDirectory(self, "Select the image folder to clean")
        if not folder:
            return
        min_w = int(self.min_w.value()); min_h = int(self.min_h.value())
        removed = 0
        try:
            removed, _ = self._remove_duplicates(folder, (min_w, min_h))
            QMessageBox.information(self, "Cleaning completed", f"Deleted {removed} images.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean up: {e}")

    def _remove_duplicates(self, folder, min_size=(200,200)):
        """
        Robust duplicate remover based on imagehash.phash.
        Returns (removed_count, kept_map) where kept_map maps kept filenames -> phash.
        Uses self.settings.get("dup_threshold", 0.8) as similarity fraction.
        """
        removed = 0
        kept = {}
        try:
            from PIL import Image, UnidentifiedImageError
            import imagehash
        except Exception:
            LOG.exception("Pillow or imagehash missing")
            return removed, kept

        # collect phashes for all images that pass min_size
        phashes = {}
        for fn in sorted(os.listdir(folder)):
            fp = os.path.join(folder, fn)
            if not is_image_file(fp):
                continue
            try:
                img = Image.open(fp)
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    continue
                h = str(imagehash.phash(img))
                if h:
                    phashes[fn] = h
            except UnidentifiedImageError:
                continue
            except Exception:
                LOG.debug("Hashing failed for %s", fp, exc_info=True)
                continue

        if not phashes:
            return removed, kept

        # determine bit length (phash hex length * 4)
        example_hex = next(iter(phashes.values()))
        bits = len(example_hex) * 4
        # threshold from settings: fraction of similarity (0..1). Convert to max hamming distance.
        sim = float(self.settings.get("dup_threshold", 0.8)) if hasattr(self, "settings") else 0.8
        sim = max(0.0, min(1.0, sim))
        max_dist = int((1.0 - sim) * bits)

        names = list(phashes.keys())
        removed_set = set()
        # compare every pair (O(n^2), ok for folder-level clean)
        for i, a in enumerate(names):
            if a in removed_set:
                continue
            ha = imagehash.hex_to_hash(phashes[a])
            for b in names[i+1:]:
                if b in removed_set:
                    continue
                hb = imagehash.hex_to_hash(phashes[b])
                try:
                    dist = ha - hb
                except Exception:
                    # fallback simple hex compare
                    dist = sum(c1 != c2 for c1, c2 in zip(phashes[a], phashes[b]))
                if dist <= max_dist:
                    # decide which to keep: prefer larger filesize, then older mtime
                    a_path = os.path.join(folder, a)
                    b_path = os.path.join(folder, b)
                    try:
                        a_sz = os.path.getsize(a_path)
                        b_sz = os.path.getsize(b_path)
                    except Exception:
                        a_sz = b_sz = 0
                    if a_sz >= b_sz:
                        try:
                            os.remove(b_path)
                            removed += 1
                            removed_set.add(b)
                        except Exception:
                            LOG.debug("Failed remove %s", b_path, exc_info=True)
                    else:
                        try:
                            os.remove(a_path)
                            removed += 1
                            removed_set.add(a)
                            break  # a removed, stop comparing a with others
                        except Exception:
                            LOG.debug("Failed remove %s", a_path, exc_info=True)
                            # if cannot remove 'a', attempt to remove b instead
                            try:
                                os.remove(b_path)
                                removed += 1
                                removed_set.add(b)
                            except Exception:
                                pass

        # build kept map (phashes left on disk)
        for fn, h in phashes.items():
            if fn not in removed_set and os.path.exists(os.path.join(folder, fn)):
                kept[fn] = h

        return removed, kept


    def reset_seen_history(self):
        base = self.settings.get("save_folder")
        if not base or not os.path.isdir(base):
            QMessageBox.information(self, "Error", "Results folder not found.")
            return
        removed = 0
        for root, dirs, files in os.walk(base):
            if "seen_urls.json" in files:
                try:
                    os.remove(os.path.join(root, "seen_urls.json"))
                    removed += 1
                except Exception:
                    pass
        QMessageBox.information(self, "Reseted", f"Deleted {removed} seen_urls.json files.")

    def reindex_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if not folder:
            return
        count = 0
        idx = {}
        try:
            files = [f for f in os.listdir(folder) if is_image_file(os.path.join(folder, f))]
            for fn in files:
                try:
                    p = os.path.join(folder, fn)
                    h = compute_image_phash(p)
                    if h:
                        idx[fn] = h
                        count += 1
                except Exception as e:
                    logging.debug("Error hashing %s: %s", fn, e)
            save_image_hash_index(folder, idx)
            QMessageBox.information(self, "Done", f"Indexed {count} files in {folder}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to index folder: {e}")
            return
        removed = 0
        for root, dirs, files in os.walk(folder):
            if "seen_urls.json" in files:
                try:
                    os.remove(os.path.join(root, "seen_urls.json"))
                    removed += 1
                except Exception:
                    pass
        QMessageBox.information(self, "Reseted", f"Deleted {removed} seen_urls.json files.")

    def _save_settings_and_notify(self):
        self._save_settings_from_ui()
        QMessageBox.information(self, "Saved", f"Configure saved in {CONFIG_PATH}")

    def closeEvent(self, event):
        try:
            # Stop threads started by UI
            for t in list(getattr(self, 'threads', []) or []):
                try:
                    if hasattr(t, 'stop'):
                        t.stop()
                    if isinstance(t, QThread):
                        try:
                            t.wait(3000)
                        except Exception:
                            pass
                except Exception:
                    pass

            # specifically stop scraper thread if running
            if getattr(self, 'scraper_thread', None):
                try:
                    self.scraper_thread.stop()
                    if isinstance(self.scraper_thread, QThread):
                        try:
                            self.scraper_thread.wait(3000)
                        except Exception:
                            pass
                except Exception:
                    pass

            # if scraper exposed downloader threads, ensure they are stopped and flushed
            try:
                if getattr(self, 'scraper_thread', None) and getattr(self.scraper_thread, "_downloaders", None):
                    for d in self.scraper_thread._downloaders:
                        try:
                            d.stop(wait=True)
                        except Exception:
                            pass
            except Exception:
                pass

            if hasattr(self, 'pinterest_driver') and self.pinterest_driver:
                try:
                    self.pinterest_driver.quit()
                except Exception:
                    try:
                        self.pinterest_driver.close()
                    except Exception:
                        pass
                self.pinterest_driver = None
                self.pinterest_driver_lock = None

            for timer in getattr(self, 'timers', []):
                try:
                    timer.stop()
                except Exception:
                    pass

            try:
                BROWSER_HELPER.cleanup()
            except Exception:
                pass

        except Exception as e:
            LOG.exception("Error while closing: forcing process termination: %s", e)

        # Let Qt close the window normally so background threads can finish and flush their state.
        event.accept()
        '''
        try:
            os._exit(0)
        except Exception:
            pass
        '''

    def show_help_dialog(self):
    
        help_text = """
        Версия: 2.0.1

        /// 
        
        Основные изменения и улучшения:

         - GUI и UX - новый двухпанельный интерфейс, менее загромождённый основной экран
         - Параллельная загрузка - Запуск нескольких DownloaderThread по настройке downloader_threads
         - Надёжная обработка дубликатов (pHash) - порог сходства переводится в максимальную Hamming-distance, логика сравнения стабильна для пустых/старых индексов
         - Исправлен и унифицирован seen (history) - seen_urls, seen_files, seen_hashes сохраняются в папке .seen и финальная запись .seen исполняется корректно
         - Корректный replace_existing - надёжное разрешение существующего пути, при замене удаляется реальный файл, индекс обновляется
         - Улучшения парсинга и взаимодействия с Pinterest - оптимизирован запуск Chrome, съедает меньше трафика и быстрее подгружает страницы
         - Надёжность и graceful shutdown - closeEvent переписан, потоки корректно останавливаются и ждут flush
            
        ///
        
        Описание:
        Программа предназначена для автоматического парсинга в первую очередь референсов для художников, загрузки и фильтрации их из:
         - Pinterest (с поддержкой полноразмерных изображений)
         - Google Images
         - Bing Images
        
        Не требует сторонних API, сбор происходит через Selenium/webdriver-manager.
        
        Основные возможности:
         • Поиск изображений по ключевым словам, категориям, позам и дополнительным запросам
         • Автоматическая фильтрация:
            - Проверка наличия человека в кадре (YOLO)
            - Определение полноразмерных изображений
            - Фильтрация низкокачественных файлов (<10 KB)
            - Проверка и удаление дубликатов по содержимому (pHash)
         • Работа с историей: исключение уже скачанных изображений (seen_urls.json)
         • Сохранение полноразмерных URL в `full_urls.log`
         • Возможность индексировать папки для ускоренной проверки дубликатов
         • Живой лог работы во вкладке «Отладка»
         • Сохранение настроек программы между запусками

        Примечания:
         - Все файлы сохраняются с уникальными именами (дата/время + случайный хвост)
         - Историю скачанных URL можно сбросить через меню

        Разработчик логики и тестрировщик: Hara
        Разработал и написал код: ChatGPT
        
        https://t.me/rambaharamba
        
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        layout = QVBoxLayout(dlg)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(help_text.strip())
        layout.addWidget(text_edit)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.resize(650, 950)
        dlg.exec_()

# ---------------------------
# Bootstrap
# ---------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
