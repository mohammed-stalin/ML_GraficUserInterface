"""Main window: sidebar navigation, page container and status bar."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ..core.dataset import Dataset
from . import theme
from .pages.classify import ClassifyPage
from .pages.cluster import ClusterPage
from .pages.data import DataPage
from .pages.export import ExportPage
from .pages.home import HomePage
from .pages.prepare import PreparePage
from .pages.regress import RegressionPage
from .pages.visualize import VisualizePage
from .widgets import TaskRunner

APP_DIR = Path(__file__).resolve().parents[2]

NAVIGATION = [
    ("home", "Home", HomePage),
    ("data", "Data", DataPage),
    ("prepare", "Prepare", PreparePage),
    ("visualize", "Visualize", VisualizePage),
    ("classify", "Classify", ClassifyPage),
    ("cluster", "Cluster", ClusterPage),
    ("regress", "Regression", RegressionPage),
    ("export", "Export", ExportPage),
]
SECTIONS = {"home": "WORKSPACE", "classify": "MODELS", "export": "OUTPUT"}
FILETYPES = [("Data files", "*.csv *.tsv *.txt *.xlsx *.xls"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"),
             ("All files", "*.*")]


class StudioApp(ctk.CTk):
    def __init__(self, appearance: str = "System"):
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme("blue")
        super().__init__(fg_color=theme.BG)
        self.title("ML Studio")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.app_dir = APP_DIR
        self.dataset: Dataset | None = None
        self.results = {"classification": None, "regression": None, "clustering": None}
        self.tasks = TaskRunner(self, on_busy=self._set_busy)
        self._status_job = None
        theme.apply_ttk_style(self)
        theme.apply_mpl_style()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._appearance_choice = appearance
        self._build_sidebar()
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)
        self.pages = {}
        for key, _label, cls in NAVIGATION:
            page = cls(main, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
        self._build_statusbar(main)
        self.current = None
        self.show_page("home")
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self):
        bar = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color=theme.SIDEBAR, border_width=0)
        bar.grid(row=0, column=0, sticky="nsw")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)
        logo = ctk.CTkFrame(bar, fg_color="transparent")
        logo.grid(row=0, column=0, sticky="ew", padx=20, pady=(26, 18))
        ctk.CTkLabel(logo, text="●", font=theme.font(22), text_color=theme.ACCENT).pack(side="left")
        ctk.CTkLabel(logo, text=" ML Studio", font=theme.font(20, "bold"), text_color=theme.TEXT).pack(side="left")

        self.nav_buttons = {}
        row = 1
        for key, label, _cls in NAVIGATION:
            if key in SECTIONS:
                ctk.CTkLabel(bar, text=SECTIONS[key], font=theme.font(10, "bold"), text_color=theme.MUTED,
                             anchor="w").grid(row=row, column=0, sticky="ew", padx=24, pady=(14, 4))
                row += 1
            btn = ctk.CTkButton(bar, text=label, anchor="w", height=36, corner_radius=8, font=theme.font(13),
                                fg_color="transparent", text_color=theme.TEXT, hover_color=theme.SURFACE_ALT,
                                command=lambda k=key: self.show_page(k))
            btn.grid(row=row, column=0, sticky="ew", padx=14, pady=1)
            self.nav_buttons[key] = btn
            row += 1

        bar.grid_rowconfigure(row, weight=1)
        row += 1
        self.dataset_card = ctk.CTkFrame(bar, fg_color=theme.SURFACE_ALT, corner_radius=10)
        self.dataset_card.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 12))
        ctk.CTkLabel(self.dataset_card, text="DATASET", font=theme.font(10, "bold"), text_color=theme.MUTED,
                     anchor="w").pack(fill="x", padx=12, pady=(10, 0))
        self.dataset_name = ctk.CTkLabel(self.dataset_card, text="None loaded", font=theme.font(12, "bold"),
                                         text_color=theme.TEXT, anchor="w", wraplength=180, justify="left")
        self.dataset_name.pack(fill="x", padx=12)
        self.dataset_shape = ctk.CTkLabel(self.dataset_card, text="Open a file to begin", font=theme.font(11),
                                          text_color=theme.MUTED, anchor="w")
        self.dataset_shape.pack(fill="x", padx=12, pady=(0, 10))
        row += 1
        self.appearance = ctk.CTkSegmentedButton(bar, values=["Light", "Dark", "System"],
                                                 command=self.set_appearance, font=theme.font(11))
        self.appearance.set(self._appearance_choice)
        self.appearance.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 18))

    def _build_statusbar(self, master):
        bar = ctk.CTkFrame(master, fg_color=theme.SURFACE, corner_radius=0, height=34, border_width=1,
                           border_color=theme.BORDER)
        bar.grid(row=1, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="Ready", font=theme.font(12), text_color=theme.MUTED, anchor="w")
        self.status.grid(row=0, column=0, sticky="ew", padx=16, pady=4)
        self.progress = ctk.CTkProgressBar(bar, width=160, mode="indeterminate", height=6)
        self.progress.grid(row=0, column=1, padx=16)
        self.progress.grid_remove()

    # -------------------------------------------------------------- navigation
    def show_page(self, key: str) -> None:
        for name, btn in self.nav_buttons.items():
            active = name == key
            btn.configure(fg_color=theme.ACCENT_SOFT if active else "transparent",
                          text_color=theme.ACCENT if active else theme.TEXT,
                          font=theme.font(13, "bold" if active else "normal"))
        page = self.pages[key]
        page.tkraise()
        page.on_show()
        self.current = key

    # -------------------------------------------------------------- data flow
    def open_dataset(self, path: str | None = None) -> None:
        path = path or filedialog.askopenfilename(title="Open dataset", filetypes=FILETYPES)
        if not path:
            return
        self.tasks.run(f"Loading {Path(path).name}…", Dataset.load, path,
                       on_done=self._dataset_loaded, on_error=lambda e: self.notify(f"Could not load file: {e}", "error"))

    def _dataset_loaded(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.results = {"classification": None, "regression": None, "clustering": None}
        self.data_changed()
        self.notify(f"Loaded {dataset.name}: {len(dataset.df):,} rows × {len(dataset.df.columns)} columns.", "success")
        self.show_page("data")

    def data_changed(self) -> None:
        """Mark every page stale and refresh the visible one."""
        ds = self.dataset
        if ds is not None:
            self.dataset_name.configure(text=ds.name)
            self.dataset_shape.configure(text=f"{len(ds.df):,} rows · {len(ds.df.columns)} columns"
                                         + (f" · {len(ds.log)} steps" if ds.log else ""))
        for page in self.pages.values():
            page.dirty = True
        if self.current:
            self.pages[self.current].on_show()

    # ------------------------------------------------------------ status/theme
    def notify(self, message: str, kind: str = "info") -> None:
        colors = {"info": theme.MUTED, "success": theme.SUCCESS, "warning": theme.WARNING, "error": theme.DANGER}
        prefix = {"success": "✓  ", "warning": "!  ", "error": "✕  "}.get(kind, "")
        self.status.configure(text=prefix + message, text_color=colors.get(kind, theme.MUTED))
        if self._status_job:
            self.after_cancel(self._status_job)
            self._status_job = None
        if kind in ("info", "success"):
            self._status_job = self.after(10000, lambda: self.status.configure(text="Ready", text_color=theme.MUTED))

    def _set_busy(self, busy: bool, message: str) -> None:
        if busy:
            self.progress.grid()
            self.progress.start()
            self.status.configure(text=message, text_color=theme.MUTED)
            self.configure(cursor="watch")
        else:
            self.progress.stop()
            self.progress.grid_remove()
            self.configure(cursor="")

    def set_appearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)
        self.after(50, self._restyle)

    def _restyle(self):
        theme.apply_ttk_style(self)
        theme.apply_mpl_style()
        for page in self.pages.values():
            page.refresh_theme()

    def close(self):
        self.tasks.shutdown()
        self.quit()
        self.destroy()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ML Studio desktop application")
    parser.add_argument("dataset", nargs="?", help="CSV or Excel file to open on start-up")
    parser.add_argument("--theme", choices=["Light", "Dark", "System"], default="System")
    args = parser.parse_args(argv)
    app = StudioApp(args.theme)
    if args.dataset:
        app.after(200, lambda: app.open_dataset(args.dataset))
    app.mainloop()
