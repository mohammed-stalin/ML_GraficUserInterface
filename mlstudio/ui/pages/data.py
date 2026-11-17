"""Dataset overview: summary tiles, preview, column profile and statistics."""
from __future__ import annotations

import customtkinter as ctk

from .. import theme
from ..widgets import DataTable, tile_row
from .base import Page, secondary_button


class DataPage(Page):
    title = "Data"
    subtitle = "Inspect the loaded dataset before preparing it."

    def build(self, body):
        secondary_button(self.actions, "Open another file…", self.app.open_dataset, height=36).pack()
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)
        self.tiles = tile_row(body, [("Rows", ""), ("Columns", ""), ("Missing cells", ""), ("Duplicate rows", "")])
        self.tiles.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.tabs = ctk.CTkTabview(body, fg_color=theme.SURFACE, segmented_button_selected_color=theme.ACCENT,
                                   corner_radius=14, border_width=1, border_color=theme.BORDER, anchor="w")
        self.tabs.grid(row=1, column=0, sticky="nsew")
        self.tables = {}
        for name in ("Preview", "Columns", "Statistics"):
            tab = self.tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            table = DataTable(tab, height=16)
            table.grid(row=0, column=0, sticky="nsew")
            self.tables[name] = table

    def refresh(self):
        ds = self.app.dataset
        s = ds.summary()
        rows, cols, missing, dupes = self.tiles.tiles
        rows.set(f"{s['rows']:,}", ds.name)
        cols.set(str(s["columns"]), f"{s['numeric_columns']} numeric · {s['categorical_columns']} text")
        cells = max(s["rows"] * s["columns"], 1)
        missing.set(f"{s['missing_cells']:,}", f"{s['missing_cells'] / cells:.1%} of all cells")
        dupes.set(f"{s['duplicate_rows']:,}", "identical rows")
        self.tables["Preview"].show(ds.df)
        self.tables["Columns"].show(ds.profile())
        desc = ds.describe()
        if desc.empty:
            self.tables["Statistics"].clear()
            self.tables["Statistics"].caption.configure(text="No numeric columns.")
        else:
            self.tables["Statistics"].show(desc)

    def refresh_theme(self):
        for table in self.tables.values():
            table.refresh_theme()
        if self.app.dataset is not None:
            self.dirty = True
            if self.app.current == "data":
                self.on_show()
