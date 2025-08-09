import os
import sys
import json
import time
import random
import logging

#LOGGING
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject


from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus
import re

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError, ImageStat

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QProgressBar, QCheckBox, QGridLayout, QGroupBox,
    QTabWidget, QComboBox, QFileDialog, QMessageBox, QAction, QLineEdit, QTextEdit, QDoubleSpinBox
)

#PSUTIL иначе хуита полностью не закроется и останется в памяти
import psutil


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spawned_pids = []

    def launch_browser(self):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        # Сохраняем PID chromedriver
        if driver.service.process:
            self.spawned_pids.append(driver.service.process.pid)
        # Сохраняем PID самого Chrome (дети chromedriver)
        try:
            p = psutil.Process(driver.service.process.pid)
            for child in p.children(recursive=True):
                self.spawned_pids.append(child.pid)
        except psutil.NoSuchProcess:
            pass
        self.driver = driver

    def closeEvent(self, event):
        # Закрываем драйвер
        try:
            self.driver.quit()
        except:
            pass
        # Убиваем все сохранённые PID
        for pid in self.spawned_pids:
            try:
                p = psutil.Process(pid)
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        # Делаем жёсткий kill для упрямых
        for pid in self.spawned_pids:
            try:
                p = psutil.Process(pid)
                if p.is_running():
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        super().closeEvent(event)


#IMAGEHASH

import threading
try:
    import imagehash
except Exception:
    imagehash = None
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#DIALOGUE

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

# icrawler (Google/Bing)
try:
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
except Exception:
    GoogleImageCrawler = None
    BingImageCrawler = None
    logging.warning("icrawler not available; Google/Bing will be disabled if missing.")

from icrawler.downloader import ImageDownloader
class CustomNameDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        from datetime import datetime
        # default_ext может быть ".jpg" или "jpg" — нормализуем
        if not default_ext.startswith("."):
            ext = f".{default_ext}"
        else:
            ext = default_ext
        return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random_nonce(6)}{ext}"

# YOLO support optional
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    YOLO = None
    HAS_YOLO = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Qt logging handler — отправляет записи логов в GUI через сигнал
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            # emit to connected Qt slot
            self.log_signal.emit(msg)
        except Exception:
            # защищённо — не ломаем основной поток логов
            pass

# Defaults
DEFAULT_SAVE_FOLDER = str(Path.cwd() / "icrawl_results")
DEFAULT_SETTINGS = {
    "save_folder": DEFAULT_SAVE_FOLDER,
    "count_per_category": 30,
    "pages_per_task": 3,
    "imgs_per_page": 10,
    "feeder_threads": 1,
    "downloader_threads": 4,
    "min_size": (200, 200),
    "pinterest_headless": True,
    # new defaults
    "use_yolo": False,
    "yolo_conf": 0.45,
    "yolo_crop": False,
    "yolo_model": "yolov8n.pt",
    "full_body_only": False,
    "full_body_ratio": 0.8,
    "only_bw": False,
    "custom_queries": "",
    "use_custom_only": False,
    "append_custom": False,
    "include_domains": "",
    "exclude_domains": ""
}

# Settings file
CONFIG_PATH = Path.home() / ".icrawl_gui_settings.json"

# Simple pose vocabulary (kept same)
POSE_TERMS = [
    "standing", "sitting", "kneeling", "crouching", "lying down",
    "walking", "running", "jumping", "stretching", "dancing",
    "foreshortening", "low angle perspective", "high angle perspective",
    "worm's eye view", "overhead view", "three quarter view", "side view",
    "front view", "back view", "dynamic action", "fighting stance",
    "attack pose", "martial arts pose", "yoga pose", "balance pose",
    "falling pose", "contrapposto", "gesture line", "silhouette study",
    "thumbnail sketch", "figure study", "anatomy breakdown", "value study",
    "line of action", "composition study", "negative space pose"
]

# Helpers ---------------------------------------------------------------------

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
    return path

def is_image_file(path):
    return os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

#FULLIMGPINTEREST
def parse_srcset_get_largest(srcset: str):
    """Возвращает URL с наибольшим разрешением из srcset (или None)."""
    if not srcset:
        return None
    # srcset: "url1 236w, url2 474w, url3 736w"
    parts = [p.strip() for p in srcset.split(',') if p.strip()]
    best = None
    best_w = 0
    for p in parts:
        seg = p.rsplit(' ', 1)
        if len(seg) == 2:
            url, w = seg
            try:
                wv = int(w.rstrip('w'))
            except Exception:
                wv = 0
            if wv > best_w:
                best_w = wv
                best = url.strip()
        else:
            # только URL без суффикса
            url = seg[0]
            if not best:
                best = url.strip()
    return best

def get_full_image_from_pin(driver, pin_href, timeout=4.0):
    """
    Открывает pin_href во временной вкладке и пытается извлечь больший (full-res)
    URL картинки — ищем meta[property='og:image'] или largest src/srcset на странице.
    Возвращает URL или None. Не уничтожает driver.
    """
    if not driver:
        return None
    try:
        current = driver.current_window_handle
    except Exception:
        current = None

    try:
        # открыть в новой вкладке и перейти туда
        driver.execute_script("window.open(arguments[0], '_blank');", pin_href)
        WebDriverWait(driver, timeout).until(lambda d: len(d.window_handles) >= 1)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(0.8)

        # попробуем meta og:image
        try:
            meta = driver.find_element("xpath", "//meta[@property='og:image']")
            if meta:
                content = meta.get_attribute("content")
                if content:
                    return content
        except Exception:
            pass

        # пробуем найти картинки и их srcset -> взять наибольший
        try:
            imgs = driver.find_elements("tag name", "img")
            best = None
            best_w = 0
            for img in imgs:
                try:
                    ss = img.get_attribute("srcset") or ""
                    if ss:
                        cand = parse_srcset_get_largest(ss)
                        if cand:
                            # грубо оценим 'w' из srcset (если нет, пропускаем)
                            # просто возвращаем первый найденный
                            return cand
                    src = img.get_attribute("src") or ""
                    if src and "pinimg" in src:
                        # fallback
                        return src
                except Exception:
                    continue
        except Exception:
            pass
    except Exception as e:
        logging.debug("get_full_image_from_pin error: %s", e)
    finally:
        # закрой вкладку и вернись
        try:
            if driver and len(driver.window_handles) > 1:
                # закрываем последнюю вкладку, затем переключаемся на предыдущую
                driver.close()
                if current:
                    driver.switch_to.window(current)
        except Exception:
            pass
    return None
#FULLIMGPINTEREST



def random_nonce(n=6):
    import string
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=n))

def slugify(s):
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')[:80]

# Per-folder seen_urls functions ----------------------------------------------

def load_seen_urls(folder):
    #"""Load seen_urls.json from folder. Return set of URLs."""
    try:
        seen_file = os.path.join(folder, "seen_urls.json")
        if os.path.exists(seen_file):
            with open(seen_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
        return set()
    except Exception as e:
        logging.debug("load_seen_urls error for %s: %s", folder, e)
        return set()

def append_seen_urls(folder, new_urls):
    #"""Append new URLs into seen_urls.json placed inside folder."""
    try:
        seen_file = os.path.join(folder, "seen_urls.json")
        existing = set()
        if os.path.exists(seen_file):
            try:
                with open(seen_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        existing = set(data)
            except Exception:
                existing = set()
        merged = existing.union(set(new_urls or []))
        with open(seen_file, "w", encoding="utf-8") as f:
            json.dump(list(merged), f, ensure_ascii=False, indent=2)
        logging.info("append_seen_urls: saved %d urls to %s (+%d new)", len(merged), seen_file, len(merged) - len(existing))
    except Exception as e:
        logging.error("append_seen_urls error for %s: %s", folder, e)

# Query builder ---------------------------------------------------------------

def build_random_query(gender, pose, sketch_mode=False, strong_random=False):
    base = f"{gender} {pose}"
    if sketch_mode:
        base += " sketch drawing lineart reference"
    else:
        base += " photo reference"
    extras = [
        "full body","gesture drawing","dynamic","anatomy","foreshortening",
        "perspective","art reference","action pose","lighting study",
        "dramatic angle","composition","contrapposto","silhouette study",
        "line of action","figure study","thumbnail sketch","value study",
        "motion study","reference sheet","model sheet","negative space",
        "croquis","life drawing"
    ]
    if strong_random:
        choose = random.sample(extras, k=random.randint(2, 3))
        base += " " + " ".join(choose)
    else:
        if random.random() < 0.6:
            base += " " + random.choice(extras)
    base += " " + random_nonce(4)
    return base.strip()

# YOLO wrapper ---------------------------------------------------------------

class YOLOFilter:
    def __init__(self, model_name="yolov8n.pt", conf=0.45):
        if not HAS_YOLO:
            raise RuntimeError("ultralytics not installed")
        # allow either local path or pretrained name
        self.model = YOLO(model_name)
        self.conf = conf

    def get_person_boxes(self, path):
        boxes_out = []
        try:
            results = self.model.predict(path, conf=self.conf, verbose=False)
            for r in results:
                if not hasattr(r, "boxes"):
                    continue
                try:
                    xy = getattr(r.boxes, "xyxy").tolist()
                    cls = getattr(r.boxes, "cls").tolist()
                except Exception:
                    continue
                for box, clsidx in zip(xy, cls):
                    idx = int(clsidx)
                    name = r.names[idx] if idx in r.names else ""
                    if name == "person" or name.lower() == "person" or name.lower().startswith("person"):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        boxes_out.append((x1, y1, x2, y2))
        except Exception as e:
            logging.debug("YOLO detect error: %s", e)
        return boxes_out

# Image utilities ------------------------------------------------------------

def is_grayscale_image(img, sample_size=1000, tolerance=6):
    """Detect if an image is grayscale.
    - Fast sampling: resize to small resolution then sample pixels.
    - tolerance: average absolute difference between channels allowed."""
    try:
        if img.mode in ("L", "LA"):
            return True
        # convert to RGB to be sure
        rgb = img.convert("RGB")
        # resize to speed up
        small = rgb.resize((100, 100))
        pixels = list(small.getdata())
        total_diff = 0
        n = 0
        # sample up to sample_size pixels
        step = max(1, len(pixels) // min(len(pixels), sample_size))
        for i in range(0, len(pixels), step):
            r, g, b = pixels[i]
            total_diff += abs(r - g) + abs(r - b) + abs(g - b)
            n += 1
        if n == 0:
            return False
        avg_diff = total_diff / (n * 3.0)
        return avg_diff <= tolerance
    except Exception as e:
        logging.debug("grayscale detect error: %s", e)
        return False

def load_image_hash_index(folder):
    idx_file = os.path.join(folder, "image_hashes.json")
    if os.path.exists(idx_file):
        try:
            with open(idx_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_image_hash_index(folder, index):
    idx_file = os.path.join(folder, "image_hashes.json")
    try:
        with open(idx_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def compute_image_phash(path):
    """Возвращает hex-строку pHash или None."""
    if imagehash is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        h = imagehash.phash(img)
        return str(h)  # hex
    except Exception as e:
        logging.debug("compute_image_phash error %s: %s", path, e)
        return None

def is_similar_hash(hex_a, hex_b, threshold=0.8):
    """
    hex strings -> сравниваем как ImageHash объекты, возвращаем True если похожи >= threshold.
    """
    try:
        ha = imagehash.hex_to_hash(hex_a)
        hb = imagehash.hex_to_hash(hex_b)
        dist = ha - hb  # hamming distance (int)
        max_bits = ha.hash.size  # обычно 64
        sim = 1.0 - float(dist) / float(max_bits)
        return sim, dist
    except Exception:
        # если что-то пошло не так — не считать дубликатом
        return 0.0, None


# Worker Thread ---------------------------------------------------------------

class ScraperThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    done_signal = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config or {}
        self._stop = False
        self.session = requests.Session()
        self._tmp_drivers = []
        self._tmp_drivers_lock = threading.Lock()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/115.0"
        ]
        # YOLO
        self.yolo = None
        if self.config.get("use_yolo", False) and HAS_YOLO:
            try:
                self.yolo = YOLOFilter(model_name=self.config.get("yolo_model", "yolov8n.pt"),
                                       conf=self.config.get("yolo_conf", 0.45))
            except Exception as e:
                logging.warning("YOLO init failed: %s", e)
                self.yolo = None

    def stop(self):
        self._stop = True
        # close HTTP session to unblock any ongoing requests
        try:
            self.session.close()
        except Exception:
            pass
        # try to quit any temporary webdriver instances created by this thread
        try:
            with getattr(self, '_tmp_drivers_lock', threading.Lock()):
                for _d in list(getattr(self, '_tmp_drivers', []) or []):
                    try:
                        _d.quit()
                    except Exception:
                        try:
                            _d.close()
                        except Exception:
                            pass
        except Exception:
            pass

    def _download_and_save(self, url, folder):
        """Download single url to folder; return filename or None."""
        if self._stop:
            return None
        try:
            headers = {"User-Agent": random.choice(self.user_agents)}
            r = self.session.get(url, headers=headers, timeout=15, stream=True)
            if r.status_code == 200:
                # try to infer extension
                ext = "jpg"
                ct = r.headers.get("content-type", "")
                if "png" in ct:
                    ext = "png"
                fn = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{random_nonce(5)}.{ext}"
                path = os.path.join(folder, fn)
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        if self._stop:
                            return None
                        if chunk:
                            f.write(chunk)
                return fn
        except Exception as e:
            logging.debug("Download error %s -> %s", url, e)
        return None
        
    def _download_and_save_url(self, url, folder):
        """
        Скачивает по URL и сохраняет с уникальным именем (timestamp + random + корректное расширение).
        Возвращает имя файла либо None.
        """
        if self._stop:
            return None
        try:
            headers = {"User-Agent": random.choice(self.user_agents)}
            r = self.session.get(url, headers=headers, timeout=20, stream=True)
            if r.status_code == 200:
                ct = r.headers.get("content-type", "").lower()
                # попытаемся определить расширение
                ext = ".jpg"
                if "png" in ct:
                    ext = ".png"
                elif "jpeg" in ct or "jpg" in ct:
                    ext = ".jpg"
                elif "webp" in ct:
                    ext = ".webp"
                elif "gif" in ct:
                    ext = ".gif"
                fn = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random_nonce(6)}{ext}"
                path = os.path.join(folder, fn)
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1024 * 8):
                        if self._stop:
                            return None
                        if chunk:
                            f.write(chunk)
                return fn
        except Exception as e:
            logging.debug("Download error %s -> %s", url, e)
        return None


    def _handle_new_file(self, folder, fn):
        """Handle duplicate detection and indexing for a newly saved file 'fn' inside 'folder'."""
        
        path = os.path.join(folder, fn)
        try:
            if os.path.getsize(path) < 10 * 1024:  # 10 KB
                logging.info("Low quality image removed: %s (less than 10KB)", fn)
                os.remove(path)
                return  # не продолжаем обработку
        except Exception as e:
            logging.warning("Failed to check file size %s: %s", fn, e)
            return

        # далее старая логика: pHash, проверка дубликатов и т.д.
        
        try:
            if not self.config.get("dup_detection", False):
                # still update index if imagehash missing? we skip
                return
            if imagehash is None:
                logging.info("imagehash not installed; skipping duplicate detection.")
                return
            idx = load_image_hash_index(folder)
            new_path = os.path.join(folder, fn)
            if not os.path.exists(new_path):
                return
            new_hash = compute_image_phash(new_path)
            if not new_hash:
                # cannot compute hash; just add placeholder
                idx[fn] = None
                save_image_hash_index(folder, idx)
                return
            best_sim = 0.0
            best_existing = None
            for ex_name, ex_hash in list(idx.items()):
                if not ex_hash:
                    continue
                sim, dist = is_similar_hash(new_hash, ex_hash, float(self.config.get("dup_threshold", 0.8)))
                if sim is None:
                    continue
                if sim >= float(self.config.get("dup_threshold", 0.8)) and sim > best_sim:
                    best_sim = sim
                    best_existing = ex_name
            action = self.config.get("dup_action", "delete_new")
            if best_existing:
                logging.info("Duplicate detected: %s ~ %s (sim=%.3f)", fn, best_existing, best_sim)
                if action == "delete_new":
                    try:
                        os.remove(new_path)
                        logging.info("Removed duplicate new file: %s", new_path)
                    except Exception as e:
                        logging.debug("Error removing new dup: %s", e)
                elif action == "replace_existing":
                    try:
                        existing_path = os.path.join(folder, best_existing)
                        if os.path.exists(existing_path):
                            os.remove(existing_path)
                        # rename new to existing name
                        os.rename(new_path, existing_path)
                        idx[best_existing] = new_hash
                        save_image_hash_index(folder, idx)
                        logging.info("Replaced %s with %s", best_existing, fn)
                    except Exception as e:
                        logging.debug("Error replacing existing: %s", e)
                else:
                    # keep both
                    idx[fn] = new_hash
                    save_image_hash_index(folder, idx)
            else:
                idx[fn] = new_hash
                save_image_hash_index(folder, idx)
        except Exception as e:
            logging.debug("dup handler error: %s", e)
    def _is_full_body_box(self, box, img_w, img_h, ratio_thresh=0.8, top_margin=0.18, bottom_margin=0.88):
        x1, y1, x2, y2 = box
        box_h = (y2 - y1)
        h_ratio = box_h / float(max(1, img_h))
        top_ok = (y1 <= img_h * top_margin)
        bottom_ok = (y2 >= img_h * bottom_margin)
        # Consider full body if height ratio is large and either top is near top or bottom near bottom
        return (h_ratio >= ratio_thresh) and (top_ok or bottom_ok)

    def run(self):
        tasks = self.config.get("tasks", [])
        total = len(tasks) or 1
        done = 0

        for task in tasks:
            if self._stop:
                break

            source = task.get("source")
            query = task.get("query")
            folder = task.get("folder")
            count = int(task.get("count", self.config.get("count_per_category", 30)))
            pages_to_sample = int(task.get("pages", self.config.get("pages_per_task", 3)))
            imgs_per_page = int(task.get("imgs_per_page", self.config.get("imgs_per_page", 10)))

            ensure_folder(folder)
            # load per-folder seen URLs
            seen = load_seen_urls(folder)

            self.status_signal.emit(f"{source.upper()}: {query}")
            logging.info("Task: source=%s query='%s' folder=%s count=%s", source, query, folder, count)

            # GOOGLE via icrawler
            if source == "google":
                if GoogleImageCrawler is None:
                    logging.error("GoogleImageCrawler not available.")
                else:
                    pages_list = list(range(0, 200, 50))
                    sample_pages = random.sample(pages_list, k=min(pages_to_sample, len(pages_list)))

                    target_new = count  # хотим столько новых файлов за запуск
                    downloaded_new = 0  # уже скачано новых

                    for offset in sample_pages:
                        if self._stop or downloaded_new >= target_new:
                            break
                        try:
                            page_num = offset // 50 + 1
                            self.status_signal.emit(f"Google: page {page_num}")
                            logging.info("Google crawl offset=%s", offset)

                            before_files = set(os.listdir(folder))
                            crawler = GoogleImageCrawler(
                                storage={"root_dir": folder},
                                feeder_threads=int(self.config.get("feeder_threads", 1)),
                                downloader_threads=int(self.config.get("downloader_threads", 4)),
                                downloader_cls=CustomNameDownloader
                            )
                            try:
                                crawler.crawl(
                                    keyword=query,
                                    max_num=target_new - downloaded_new,
                                    offset=offset
                                )
                            except TypeError:
                                crawler.crawl(
                                    keyword=query,
                                    max_num=target_new - downloaded_new
                                )

                            after_files = set(os.listdir(folder))
                            new_files = [
                                f for f in (after_files - before_files)
                                if is_image_file(os.path.join(folder, f))
                            ]
                            downloaded_new += len(new_files)
                            # handle newly created files: update seen and run duplicate handler
                            for _new_fn in new_files:
                                try:
                                    seen.add(_new_fn)
                                    # write seen immediately
                                    try:
                                        with open(os.path.join(folder, "seen_urls.json"), "w", encoding="utf-8") as sf:
                                            json.dump(list(seen), sf, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        logging.debug("Could not update seen_urls.json: %s", e)
                                    # run duplicate detection/indexing
                                    try:
                                        self._handle_new_file(folder, _new_fn)
                                    except Exception as e:
                                        logging.debug("Error in dup handler for %s: %s", _new_fn, e)
                                except Exception as e:
                                    logging.debug("Error processing new file %s: %s", _new_fn, e)

                        except Exception as e:
                            logging.error("Google crawl error: %s", e)
                            
                            

            # BING via icrawler
            elif source == "bing":
                if BingImageCrawler is None:
                    logging.error("BingImageCrawler not available.")
                else:
                    pages_list = list(range(0, 200, 50))
                    sample_pages = random.sample(pages_list, k=min(pages_to_sample, len(pages_list)))

                    target_new = count  # хотим столько новых файлов за запуск
                    downloaded_new = 0  # уже скачано новых

                    for offset in sample_pages:
                        if self._stop or downloaded_new >= target_new:
                            break
                        try:
                            page_num = offset // 50 + 1
                            self.status_signal.emit(f"Bing: page {page_num}")
                            logging.info("Bing crawl offset=%s", offset)

                            before_files = set(os.listdir(folder))
                            crawler = BingImageCrawler(
                                storage={"root_dir": folder},
                                feeder_threads=int(self.config.get("feeder_threads", 1)),
                                downloader_threads=int(self.config.get("downloader_threads", 4)),
                                downloader_cls=CustomNameDownloader
                            )
                            try:
                                crawler.crawl(
                                    keyword=query,
                                    max_num=target_new - downloaded_new,
                                    offset=offset
                                )
                            except TypeError:
                                crawler.crawl(
                                    keyword=query,
                                    max_num=target_new - downloaded_new
                                )

                            after_files = set(os.listdir(folder))
                            new_files = [
                                f for f in (after_files - before_files)
                                if is_image_file(os.path.join(folder, f))
                            ]
                            downloaded_new += len(new_files)
                            # handle newly created files: update seen and run duplicate handler
                            for _new_fn in new_files:
                                try:
                                    seen.add(_new_fn)
                                    # write seen immediately
                                    try:
                                        with open(os.path.join(folder, "seen_urls.json"), "w", encoding="utf-8") as sf:
                                            json.dump(list(seen), sf, ensure_ascii=False, indent=2)
                                    except Exception as e:
                                        logging.debug("Could not update seen_urls.json: %s", e)
                                    # run duplicate detection/indexing
                                    try:
                                        self._handle_new_file(folder, _new_fn)
                                    except Exception as e:
                                        logging.debug("Error in dup handler for %s: %s", _new_fn, e)
                                except Exception as e:
                                    logging.debug("Error processing new file %s: %s", _new_fn, e)

                        except Exception as e:
                            logging.error("Bing crawl error: %s", e)

            # PINTEREST via Selenium (full res)
            elif source == "pinterest":
                try:
                    from selenium import webdriver
                    from selenium.webdriver.chrome.service import Service
                    from selenium.webdriver.chrome.options import Options
                    from selenium.webdriver.common.by import By
                    from webdriver_manager.chrome import ChromeDriverManager
                except Exception as e:
                    logging.error("Selenium/webdriver-manager not available: %s", e)
                    self.status_signal.emit("Pinterest: selenium not installed")
                    time.sleep(0.2)
                    done += 1
                    self.progress_signal.emit(int(done / total * 100))
                    continue

                # опции драйвера
                chrome_options = Options()
                if self.config.get("pinterest_headless", True):
                    try:
                        chrome_options.add_argument("--headless=new")
                    except Exception:
                        chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--log-level=3")
                chrome_options.add_argument(f"user-agent={random.choice(self.user_agents)}")

                # попытаться использовать уже открытый драйвер (передан в config)
                driver = self.config.get("pinterest_driver", None)
                driver_lock = self.config.get("pinterest_driver_lock", None)

                def _acquire_driver():
                    nonlocal driver
                    # если драйвер передан и reuse включён — используем его
                    if driver:
                        return driver, driver_lock
                    # иначе создаём временный локальный драйвер (он будет закрыт в finally)
                    try:
                        service = Service(ChromeDriverManager().install())
                        tmp_driver = webdriver.Chrome(service=service, options=chrome_options)
                        try:
                            # track temporary drivers so we can quit them on stop()
                            with getattr(self, '_tmp_drivers_lock', threading.Lock()):
                                self._tmp_drivers.append(tmp_driver)
                        except Exception:
                            pass
                        return tmp_driver, None
                    except Exception as e:
                        logging.error("Failed to start Chrome for Pinterest: %s", e)
                        return None, None

                tmp_created = False
                driver_instance, lock = _acquire_driver()
                if driver_instance is None:
                    # не можем парсить Pinterest
                    done += 1
                    self.progress_signal.emit(int(done / total * 100))
                    continue

                try:
                    if driver is None:
                        tmp_created = True

                    # sample pages
                    page_choices = random.sample(range(1, 1 + max(1, pages_to_sample * 2)), k=pages_to_sample)
                    for page in page_choices:
                        if self._stop:
                            break
                        search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}&page={page}"
                        self.status_signal.emit(f"Pinterest: page {page}")
                        logging.info("Pinterest open %s", search_url)
                        try:
                            # если есть lock — захватываем перед работой с драйвером
                            if lock:
                                lock.acquire()
                            driver_instance.get(search_url)
                        except Exception as e:
                            logging.debug("Selenium get error: %s", e)
                            if lock:
                                lock.release()
                            continue
                        finally:
                            if lock and lock.locked():
                                # оставляем захваченный lock при навигации кратко (можно освободить)
                                try:
                                    lock.release()
                                except Exception:
                                    pass

                        time.sleep(1.4)
                        # скроллим
                        last_h = driver_instance.execute_script("return document.body.scrollHeight")
                        for _ in range(3):
                            if self._stop:
                                break
                            driver_instance.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(1.0)
                            new_h = driver_instance.execute_script("return document.body.scrollHeight")
                            if new_h == last_h:
                                break
                            last_h = new_h

                        # собираем элементы <img>
                        try:
                            imgs = driver_instance.find_elements(By.TAG_NAME, "img")
                        except Exception as e:
                            logging.debug("Pinterest collect error: %s", e)
                            imgs = []

                        img_urls_candidates = []
                        for img_el in imgs:
                            try:
                                srcset = img_el.get_attribute("srcset") or ""
                                src = img_el.get_attribute("src") or ""
                                # пробуем srcset -> largest
                                best = parse_srcset_get_largest(srcset)
                                if best:
                                    img_urls_candidates.append((best, None))  # (url, pin_href)
                                    continue
                                if src and "pinimg" in src:
                                    # возможно превью, но добавим как fallback
                                    # также пытаемся найти ссылку на сам pin
                                    try:
                                        anchor = img_el.find_element(By.XPATH, "./ancestor::a[1]")
                                        href = anchor.get_attribute("href") if anchor else None
                                    except Exception:
                                        href = None
                                    img_urls_candidates.append((src, href))
                            except Exception:
                                continue

                        random.shuffle(img_urls_candidates)
                        take_n = min(len(img_urls_candidates), imgs_per_page)
                        for cand, pin_href in img_urls_candidates[:take_n]:
                            if self._stop:
                                break
                            # если опция full-res включена — пробуем получить полное изображение через pin page
                            full_res_url = None
                            if self.config.get("pinterest_full_res", False) and pin_href:
                                try:
                                    if lock:
                                        lock.acquire()
                                    full_res_url = get_full_image_from_pin(driver_instance, pin_href)
                                except Exception as e:
                                    logging.debug("Error getting full-res from pin: %s", e)
                                finally:
                                    if lock and lock.locked():
                                        try:
                                            lock.release()
                                        except Exception:
                                            pass

                            # если не получилось — используем cand
                            download_url = full_res_url or cand
                            if not download_url:
                                continue

                            # логируем полный URL (в логах требуется видеть ссылки на full-res)
                            logging.info("Pinterest FULL URL: %s (pin: %s)", download_url, pin_href)

                            # domain include/exclude
                            incl = task.get("include_domains") or self.config.get("include_domains") or ""
                            excl = task.get("exclude_domains") or self.config.get("exclude_domains") or ""
                            domain = ""
                            try:
                                domain = download_url.split('/')[2]
                            except Exception:
                                domain = ""
                            if incl:
                                ok = any(d.strip().lower() in domain.lower() for d in incl.split(','))
                                if not ok:
                                    logging.debug("Skipping %s - not in include domains", download_url)
                                    continue
                            if excl:
                                if any(d.strip().lower() in domain.lower() for d in excl.split(',')):
                                    logging.debug("Skipping %s - in exclude domains", download_url)
                                    continue

                            # пропускаем если уже в seen
                            if download_url in seen:
                                continue

                            # скачиваем URL напрямую через requests (используем метод, который даёт уникальные имена)
                            fn = None
                            try:
                                fn = self._download_and_save_url(download_url, folder)
                            except Exception as e:
                                logging.debug("Download from pinterest failed %s: %s", download_url, e)
                                fn = None

                            if fn:
                                seen.add(download_url)
                                logging.info("Pinterest saved %s (from %s)", fn, download_url)
                                # write full urls log
                                try:
                                    with open(os.path.join(folder, "full_urls.log"), "a", encoding="utf-8") as lf:
                                        lf.write(download_url + "\n")
                                except Exception as e:
                                    logging.debug("Could not write full_urls.log: %s", e)
                                # update seen file with URLs
                                try:
                                    with open(os.path.join(folder, "seen_urls.json"), "w", encoding="utf-8") as sf:
                                        json.dump(list(seen), sf, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    logging.debug("Could not update seen_urls.json: %s", e)
                                # run duplicate handler on new file
                                try:
                                    self._handle_new_file(folder, fn)
                                except Exception as e:
                                    logging.debug("Error in dup handler for %s: %s", fn, e)
                                self.status_signal.emit(f"P: saved {fn}")

                finally:
                    # если мы создали временный драйвер, то закрываем
                    if tmp_created and driver_instance:
                        try:
                            driver_instance.quit()
                        except Exception:
                            pass




            
            
            try:
                if (self.config.get("use_yolo") and self.yolo) or self.config.get("only_bw") or self.config.get("min_size"):
                    crop_enabled = self.config.get("yolo_crop", False)
                    crop_folder = None
                    if crop_enabled:
                        crop_folder = os.path.join(folder, "_crops")
                        ensure_folder(crop_folder)
                    files = list(os.listdir(folder))
                    for fn in files:
                        if self._stop:
                            break
                        fpath = os.path.join(folder, fn)
                        if not is_image_file(fpath):
                            continue
                        # basic size filter
                        try:
                            img = Image.open(fpath)
                        except (UnidentifiedImageError, OSError, ValueError) as e:
                            try:
                                os.remove(fpath)
                            except Exception:
                                pass
                            continue
                        img_w, img_h = img.size
                        min_w, min_h = self.config.get("min_size", (100, 100))
                        if img_w < min_w or img_h < min_h:
                            try:
                                os.remove(fpath)
                            except Exception:
                                pass
                            continue

                        # YOLO person detection (if enabled)
                        boxes = []
                        if self.config.get("use_yolo") and self.yolo:
                            try:
                                boxes = self.yolo.get_person_boxes(fpath)
                            except Exception as e:
                                logging.debug("YOLO per-image error: %s", e)
                                boxes = []
                            # if enabled, require at least one person detected
                            if not boxes:
                                try:
                                    os.remove(fpath)
                                except Exception:
                                    pass
                                continue

                        # full-body check (if requested)
                        if self.config.get("full_body_only") and boxes:
                            keep = False
                            ratio_thresh = float(self.config.get("full_body_ratio", 0.8))
                            for box in boxes:
                                if self._is_full_body_box(box, img_w, img_h, ratio_thresh=ratio_thresh):
                                    keep = True
                                    break
                            if not keep:
                                try:
                                    os.remove(fpath)
                                except Exception:
                                    pass
                                continue

                        # black-and-white check
                        if self.config.get("only_bw"):
                            try:
                                if not is_grayscale_image(img):
                                    try:
                                        os.remove(fpath)
                                    except Exception:
                                        pass
                                    continue
                            except Exception as e:
                                logging.debug("BW check error: %s", e)
                                continue

                        # save crops if YOLO found boxes and crop enabled
                        if boxes and crop_enabled:
                            try:
                                for i, (x1, y1, x2, y2) in enumerate(boxes):
                                    crop = img.crop((x1, y1, x2, y2))
                                    crop_name = f"{Path(fn).stem}_crop{i}.jpg"
                                    crop.save(os.path.join(crop_folder, crop_name))
                                    logging.info("YOLO: saved crop %s", crop_name)
                            except Exception as e:
                                logging.debug("YOLO crop error: %s", e)
            except Exception as e:
                logging.debug("Postprocess top-level error: %s", e)

            # append per-folder seen URLs and finish task
            try:
                append_seen_urls(folder, seen)
            except Exception as e:
                logging.debug("append_seen_urls error: %s", e)

            done += 1
            try:
                self.progress_signal.emit(int(done / float(total) * 100))
            except Exception:
                self.progress_signal.emit(0)

        self.done_signal.emit()

# GUI -------------------------------------------------------------------------

class QDoubleSpinBoxWithDefaults(QDoubleSpinBox):
    def __init__(self, default=0.45):
        super().__init__()
        self.setRange(0.05, 0.95)
        self.setSingleStep(0.05)
        self.setValue(default)

class MainWindow(QMainWindow):

    def start_pinterest_browser(self, silent=True):
        # запускается в GUI потоке — создаем браузер и сохраняем + lock
        if getattr(self, 'pinterest_driver', None):
            logging.info("Pinterest driver already running.")
            if not silent:
                QMessageBox.information(self, 'Browser', 'Browser is running.')
            # ensure buttons state
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
            if self.settings.get('pinterest_headless', True):
                try:
                    opts.add_argument('--headless=new')
                except Exception:
                    opts.add_argument('--headless')
            opts.add_argument('--no-sandbox')
            opts.add_argument('--disable-dev-shm-usage')
            opts.add_argument('--log-level=3')
            # use local cache for driver to avoid repeated downloads
            driver_path = ChromeDriverManager().install()
            service = Service(driver_path)
            self.pinterest_driver = webdriver.Chrome(service=service, options=opts)
            self.pinterest_driver_lock = threading.Lock()
            logging.info('Started Pinterest driver at %s', driver_path)
            # update UI buttons
            try:
                self.pinterest_start_btn.setEnabled(False)
                self.pinterest_stop_btn.setEnabled(True)
            except Exception:
                pass
            if not silent:
                QMessageBox.information(self, 'Browser', 'Pinterest browser is running.')
        except Exception as e:
            logging.error('Failed to start Pinterest browser: %s', e)
            if not silent:
                QMessageBox.critical(self, 'Error', f'Failed to start browser: {e}')


    def stop_pinterest_browser(self):
        d = getattr(self, "pinterest_driver", None)
        if not d:
            logging.info("Pinterest driver not running (stop invoked).")
            QMessageBox.information(self, "Browser", "The browser is not running.")
            # ensure buttons reflect stopped state
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


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Scraper by Hara (modified and wrote by ChatGPT) for personal use")
        self.setGeometry(120, 120, 980, 760)
        
        # load settings from disk if available
        self.settings = DEFAULT_SETTINGS.copy()

        # --- Live log view для вкладки "Отладка" ---
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # Создаём и подключаем QtLogHandler (не дублируем, если уже добавлен)
        self._qt_log_handler = QtLogHandler()
        self._qt_log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

        # Добавляем handler к корневому логгеру только один раз
        root_logger = logging.getLogger()
        already = False
        for h in getattr(root_logger, "handlers", []):
            if isinstance(h, QtLogHandler):
                already = True
                break
        if not already:
            # подключаем сигнал к методу _append_log (метод добавим ниже)
            self._qt_log_handler.log_signal.connect(self._append_log)
            root_logger.addHandler(self._qt_log_handler)
        # --- /Live log view ---

        self._load_settings_from_disk()
        ensure_folder(self.settings["save_folder"])
        self.threads = []
        self.scraper_thread = None
        self._init_ui()
        # start pinterest driver automatically if pinterest enabled
        try:
            if self.settings.get("use_pinterest", True):
                self.start_pinterest_browser(silent=True)
        except Exception as e:
            logging.debug("Auto-start Pinterest driver failed: %s", e)
    
    def _append_log(self, msg: str):
        """Добавляет строку в виджет логов (вызов из сигнала Qt)."""
        try:
            # append + автоскролл
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
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout()
        central.setLayout(v)

        tabs = QTabWidget()
        tabs.addTab(self._tab_sources(), "Sources")
        tabs.addTab(self._tab_poses(), "Poses")
        tabs.addTab(self._tab_custom_queries(), "Additional requests")  # new tab
        tabs.addTab(self._tab_filters(), "Filters")
        tabs.addTab(self._tab_debug(), "Debug")
        v.addWidget(tabs)

        bottom = QHBoxLayout()
        self.folder_label = QLabel(f"Save: {self.settings['save_folder']}")
        bottom.addWidget(self.folder_label)
        btn_choose = QPushButton("Choose folder")
        btn_choose.clicked.connect(self.choose_folder)
        bottom.addWidget(btn_choose)
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_scraping)
        bottom.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_scraping)
        self.btn_stop.setEnabled(False)
        bottom.addWidget(self.btn_stop)
        v.addLayout(bottom)

        self.status_label = QLabel("Ready.")
        v.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        v.addWidget(self.progress_bar)

        # menu
        menubar = self.menuBar()
        tools = menubar.addMenu("Tools")
        act_clean = QAction("Clean duplicates", self)
        act_clean.triggered.connect(self.clean_duplicates_manual)
        tools.addAction(act_clean)
        act_reset_seen = QAction("Reset seen_urls (for all folders)", self)
        act_reset_seen.triggered.connect(self.reset_seen_history)
        tools.addAction(act_reset_seen)
        # save settings action
        act_save = QAction("Save configuration", self)
        act_save.triggered.connect(lambda: self._save_settings_and_notify())
        tools.addAction(act_save)
        #INFO      
        act_help = QAction("Info", self)
        act_help.triggered.connect(self.show_help_dialog)
        tools.addAction(act_help)
        

    def _tab_debug(self):
        w = QWidget()
        v = QVBoxLayout()
        w.setLayout(v)
        v.addWidget(QLabel("Parser logs:"))
        # если по какой-то причине self.log_view ещё не создан — создаём локально и присваиваем
        if not hasattr(self, "log_view") or self.log_view is None:
            self.log_view = QTextEdit()
            self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)
        return w

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
        # genders
        gb = QGroupBox("Gender"); hl = QHBoxLayout()
        self.gender_checks = {}
        saved_genders = set(self.settings.get("genders", ["man","woman","boy","girl"])) 
        for g in ["man","woman","boy","girl"]:
            cb = QCheckBox(g); cb.setChecked(g in saved_genders); self.gender_checks[g] = cb; hl.addWidget(cb)
        gb.setLayout(hl); v.addWidget(gb)
        # poses
        pbox = QGroupBox("Poses"); pl = QGridLayout()
        self.pose_checks = {}
        saved_poses = set(self.settings.get("poses", POSE_TERMS[:6]))
        cols = 3
        for idx, term in enumerate(POSE_TERMS):
            cb = QCheckBox(term); cb.setChecked(term in saved_poses); self.pose_checks[term] = cb
            pl.addWidget(cb, idx // cols, idx % cols)
        pbox.setLayout(pl); v.addWidget(pbox)
        # presets and random options
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
        # domain include/exclude basic
        v.addWidget(QLabel("Include domains (comma separated, optional) — will be saved in Pinterest task/manual scraping"))
        self.include_domains_le = QLineEdit(self.settings.get("include_domains", "")); v.addWidget(self.include_domains_le)
        v.addWidget(QLabel("Exclude domains (comma separated, optional)"))
        self.exclude_domains_le = QLineEdit(self.settings.get("exclude_domains", "")); v.addWidget(self.exclude_domains_le)
        return w

    def _tab_filters(self):
        w = QWidget(); v = QVBoxLayout(); w.setLayout(v)

        # --- Pinterest settings ---
        self.pinterest_fullres_cb = QCheckBox("Download full-res images (if available)")
        self.pinterest_fullres_cb.setChecked(bool(self.settings.get("pinterest_full_res", False)))
        v.addWidget(self.pinterest_fullres_cb)
        
        # Кнопки управления браузером
        hbr = QHBoxLayout()
        self.pinterest_start_btn = QPushButton("Start browser")
        self.pinterest_start_btn.clicked.connect(self.start_pinterest_browser)
        hbr.addWidget(self.pinterest_start_btn)

        self.pinterest_stop_btn = QPushButton("Stop browser")
        self.pinterest_stop_btn.setEnabled(False)
        self.pinterest_stop_btn.clicked.connect(self.stop_pinterest_browser)
        hbr.addWidget(self.pinterest_stop_btn)
        v.addLayout(hbr)
        # --- End Pinterest settings ---

        # --- Duplicate detection settings ---
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
        # Button to reindex folder hashes
        self.reindex_btn = QPushButton("Index folder (hashes)")
        self.reindex_btn.clicked.connect(self.reindex_folder_dialog)
        v.addWidget(self.reindex_btn)

        # --- End Duplicate detection settings ---


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
            # persist immediately
            self._save_settings_from_ui()

    def _gather_ui_settings(self):
        # gather persistent settings from UI into self.settings dict
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


        # custom queries and domain filters
        self.settings["custom_queries"] = str(self.custom_queries_edit.toPlainText() or "")
        self.settings["use_custom_only"] = bool(self.use_custom_only_cb.isChecked())
        self.settings["append_custom"] = bool(self.append_custom_cb.isChecked())
        self.settings["include_domains"] = str(self.include_domains_le.text() or "")
        self.settings["exclude_domains"] = str(self.exclude_domains_le.text() or "")

    def _save_settings_from_ui(self):
        self._gather_ui_settings()
        # also persist to disk
        self._save_settings_to_disk()


    def collect_config_and_tasks(self):
        # first update settings from UI
        self._gather_ui_settings()

        cfg = {}
        cfg.update(self.settings)  # copy everything
        sources = []
        if self.settings.get("use_google", True): sources.append("google")
        if self.settings.get("use_bing", False): sources.append("bing")
        if self.settings.get("use_pinterest", True): sources.append("pinterest")
        cfg["sources"] = sources

        tasks = []

        # handle custom queries (one per line)
        custom_raw = self.settings.get("custom_queries", "").strip()
        custom_list = [line.strip() for line in custom_raw.splitlines() if line.strip()]

        # if custom-only, create tasks from custom queries only
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
            # generate queries from genders/poses
            genders = self.settings.get("genders", [])
            poses = self.settings.get("poses", [])
            if not sources or not genders or not poses:
                # allow fallback to custom queries even if not full config
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
                    # nothing to do
                    cfg["tasks"] = []
                    return cfg

            # otherwise build generated queries (with auto-append of custom queries)
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
                            # if custom list provided, also create queries that append custom terms to this generated query
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

        # persist some settings and save to disk
        self._save_settings_from_ui()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")

        thread_cfg = cfg.copy()
        # pass the pre-initialized Pinterest driver to the worker thread (if available)
        if getattr(self, "pinterest_driver", None):
            thread_cfg["pinterest_driver"] = self.pinterest_driver
            thread_cfg["pinterest_driver_lock"] = self.pinterest_driver_lock

        

        thread_cfg.update(self.settings)
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
            self.status_label.setText("Waiting for scaping to stop...")

    def on_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Completed.")
        QMessageBox.information(self, "Done", "Parsing complete.")
        # save settings when finished
        self._save_settings_from_ui()
        # cleanup thread registry
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
        removed = 0
        kept = {}
        for fn in list(os.listdir(folder)):
            fp = os.path.join(folder, fn)
            if not is_image_file(fp):
                continue
            try:
                img = Image.open(fp)
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    os.remove(fp); removed += 1
            except (UnidentifiedImageError, OSError, ValueError):
                try:
                    os.remove(fp); removed += 1
                except Exception:
                    pass
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
        # Выбор папки для индексирования
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if not folder:
            return

        # Индексация изображений
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

        # Удаление всех seen_urls.json в выбранной папке и подпапках
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
                # First, attempt to stop background QThreads gracefully
                try:
                    for t in list(getattr(self, 'threads', []) or []):
                        try:
                            if hasattr(t, 'stop'):
                                t.stop()
                            # wait briefly for thread to finish
                            try:
                                t.wait(3000)
                            except Exception:
                                pass
                            if t.isRunning():
                                try:
                                    t.terminate()
                                except Exception:
                                    pass
                                try:
                                    t.wait()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
        
                # Закрыть Selenium WebDriver (GUI-owned)
                if hasattr(self, 'pinterest_driver') and self.pinterest_driver:
                    try:
                        self.pinterest_driver.quit()
                    except Exception:
                        try:
                            self.pinterest_driver.close()
                        except Exception:
                            pass
        
                    # give a bit of time for subprocesses to terminate
                    try:
                        time.sleep(0.5)
                    except Exception:
                        pass
        
                    self.pinterest_driver = None
                    self.pinterest_driver_lock = None
        
                # Остановить все таймеры
                for timer in getattr(self, 'timers', []):
                    try:
                        timer.stop()
                    except Exception:
                        pass
        
            except Exception as e:
                print('Error while closing: gonna try to force terminating the entire process', e)
        
            event.accept()
        
            # Принудительное завершение всего процесса (последний шанс)
            try:
                import sys, os
                os._exit(0)
            except Exception:
                pass

    def show_help_dialog(self):
            help_text = """
            
            RU
            Версия: 1.0.1

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
            
            EN 
            Version: 1.0.1
            
            Description:
            The program is designed for automatic parsing primarily references for artists, downloading and filtering them from:
            - Pinterest (with support for full-size images)
            - Google Images
            - Bing Images

            Does not require third-party APIs, collection occurs through Selenium/webdriver-manager.

            Main features:
            • Search images by keywords, categories, poses and additional queries
            • Automatic filtering:
            - Check for a person in the frame (YOLO)
            - Detection of full-size images
            - Filtering low-quality files (<10 KB)
            - Checking and removing duplicates by content (pHash)
            • Working with history: excluding already downloaded images (seen_urls.json)
            • Saving full-size URLs in `full_urls.log`
            • Ability to index folders for faster duplicate checking
            • Live log of work in the "Debug" tab
            • Saving program settings between launches

            Notes:
            - All files are saved with unique names (date/time + random tail)
            - The history of downloaded URLs can be reset via the menu

            Logic developer and tester: Hara
            Developed and wrote the code: ChatGPT
            
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


# Run -------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec_())