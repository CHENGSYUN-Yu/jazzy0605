import os
import sys
import threading
import time
import tkinter as tk

import dearpygui.dearpygui as dpg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.app import App

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_reload_flag = threading.Event()


class _SrcWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            _reload_flag.set()


def _get_screen_size() -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def _open_style_editor():
    if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
        dpg.show_style_editor()


def main():
    screen_w, screen_h = _get_screen_size()

    # Live reload：監控 src/ 和 main.py
    observer = Observer()
    observer.schedule(_SrcWatcher(), path="src", recursive=True)
    observer.schedule(_SrcWatcher(), path=".", recursive=False)
    observer.start()

    dpg.create_context()

    with dpg.font_registry(tag="font_registry"):
        with dpg.font(_FONT_PATH, 36, tag="font_36"):
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
    dpg.bind_font("font_36")

    # Ctrl+E → Style Editor
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_E, callback=_open_style_editor)

    app = App(screen_w=screen_w, screen_h=screen_h)
    app.setup()

    dpg.create_viewport(title="Kassow RobotUse  [Ctrl+E=Style Editor | 存檔=自動重載]",
                        width=screen_w, height=screen_h)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()

    _frame_dt = 1.0 / 30  # 30 FPS 上限
    while dpg.is_dearpygui_running():
        _t0 = time.perf_counter()
        if _reload_flag.is_set():
            print("\n[Live Reload] 偵測到檔案變動，重新啟動…")
            break
        app.update_camera()
        dpg.render_dearpygui_frame()
        _elapsed = time.perf_counter() - _t0
        if _elapsed < _frame_dt:
            time.sleep(_frame_dt - _elapsed)

    app.cleanup()
    observer.stop()
    observer.join()
    dpg.destroy_context()

    if _reload_flag.is_set():
        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
