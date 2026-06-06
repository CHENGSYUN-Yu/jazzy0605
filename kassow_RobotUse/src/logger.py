import datetime
import os
import queue
import threading


class Logger:
    """背景寫檔 + GUI 同步顯示的記錄器。

    使用方式：
        logger = Logger(log_dir="logs")
        logger.bind_gui("ros2_log")   # 綁定 dpg item tag（可選）
        logger.log("[INFO] 連線成功")
    """

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._filepath = os.path.join(log_dir, f"{ts}.txt")
        self._gui_tag: str | None = None
        self._q: queue.SimpleQueue[str] = queue.SimpleQueue()
        threading.Thread(target=self._writer_loop, daemon=True).start()
        self._write_raw(f"=== Kassow RobotUse 啟動 {ts} ===")

    def bind_gui(self, tag: str) -> None:
        """綁定 dearpygui multiline input tag，log 時同步顯示。"""
        self._gui_tag = tag

    def log(self, msg: str) -> None:
        """記錄一筆訊息（時間戳自動加上）。"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._write_raw(line)
        self._update_gui(line)

    def _write_raw(self, line: str) -> None:
        self._q.put(line)

    def _update_gui(self, line: str) -> None:
        if not self._gui_tag:
            return
        try:
            import dearpygui.dearpygui as dpg
            if dpg.does_item_exist(self._gui_tag):
                dpg.set_value(self._gui_tag,
                              dpg.get_value(self._gui_tag) + line + "\n")
        except Exception:
            pass

    def _writer_loop(self) -> None:
        with open(self._filepath, "w", encoding="utf-8", buffering=1) as f:
            while True:
                f.write(self._q.get() + "\n")

    @property
    def filepath(self) -> str:
        return self._filepath
