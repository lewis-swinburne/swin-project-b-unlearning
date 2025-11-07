# src/utils/monitor.py
import psutil, csv, time, threading, os
from datetime import datetime

class ResourceLogger:
    def __init__(self, outfile: str, interval_sec: float = 1.0):
        self.outfile = outfile
        self.interval = interval_sec
        self._stop = threading.Event()
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    def _loop(self):
        with open(self.outfile, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts","cpu_percent","ram_mb"])
            proc = psutil.Process()
            while not self._stop.is_set():
                cpu = psutil.cpu_percent(interval=None)
                rss_mb = proc.memory_info().rss / (1024*1024)
                w.writerow([datetime.now().isoformat(timespec="seconds"), cpu, f"{rss_mb:.1f}"])
                f.flush()
                time.sleep(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._t.join(timeout=2)
