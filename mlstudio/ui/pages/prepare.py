"""Data preparation with column selection, operation cards and undo history."""
from __future__ import annotations

import customtkinter as ctk

from ...core import dataset as ops
from .. import theme
from ..widgets import ColumnPicker, DataTable
from .base import Page, control_label, secondary_button

FILL_STRATEGIES = {"Mean (numeric) / mode (text)": "mean", "Median (numeric) / mode (text)": "median",
                   "Most frequent value": "mode", "Zero (numeric) / mode (text)": "zero"}


class PreparePage(Page):
    title = "Prepare"
    subtitle = "Select columns on the left, then apply an operation. Every step can be undone."

    def build(self, body):
        self.undo_btn = secondary_button(self.actions, "Undo", self.undo, width=90, height=36)
        self.undo_btn.pack(side="left", padx=(0, 8))
        secondary_button(self.actions, "Reset to original", self.reset, height=36).pack(side="left")

        body.grid_columnconfigure(2, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=14, border_width=1, border_color=theme.BORDER,
                            width=300)
        left.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(left, text="Columns", font=theme.font(15, "bold"), text_color=theme.TEXT,
                     anchor="w").pack(fill="x", padx=16, pady=(14, 2))
        ctk.CTkLabel(left, text="Empty selection = all columns (where it applies).", font=theme.font(11),
                     text_color=theme.MUTED, anchor="w").pack(fill="x", padx=16)
        self.picker = ColumnPicker(left, height=380)
        self.picker.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        ops_frame = ctk.CTkScrollableFrame(body, fg_color="transparent", width=330)
        ops_frame.grid(row=0, column=1, sticky="nsew", padx=14)
        ops_frame.grid_columnconfigure(0, weight=1)
        self._ops_section(ops_frame, 0, "Missing values", [
            ("fill", "Fill missing values"), ("dropna", "Drop rows with missing values")], extra=self._fill_menu)
        self._ops_section(ops_frame, 1, "Encoding", [
            ("onehot", "One-hot encode"), ("ordinal", "Ordinal encode (integer codes)")])
        self._ops_section(ops_frame, 2, "Rows & columns", [
            ("drop", "Drop selected columns"), ("dupes", "Remove duplicate rows"),
            ("outliers", "Remove outliers (1.5 × IQR)"), ("scale", "Standardize (z-score)")])
        self._ops_section(ops_frame, 3, "History", [])
        self.history = ctk.CTkLabel(self._last_card, text="No steps yet.", font=theme.font(12),
                                    text_color=theme.MUTED, anchor="w", justify="left", wraplength=270)
        self.history.pack(fill="x")

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        control_label(right, "PREVIEW").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.table = DataTable(right, max_rows=300, height=20)
        self.table.grid(row=1, column=0, sticky="nsew")

    def _fill_menu(self, master):
        self.fill_strategy = ctk.CTkOptionMenu(master, values=list(FILL_STRATEGIES), height=32)
        self.fill_strategy.pack(fill="x", pady=(0, 6))

    def _ops_section(self, master, row, title, buttons, extra=None):
        card = ctk.CTkFrame(master, fg_color=theme.SURFACE, corner_radius=14, border_width=1, border_color=theme.BORDER)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(card, text=title, font=theme.font(14, "bold"), text_color=theme.TEXT,
                     anchor="w").pack(fill="x", padx=16, pady=(12, 8))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 14))
        if extra:
            extra(inner)
        for key, label in buttons:
            ctk.CTkButton(inner, text=label, height=34, anchor="w", command=lambda k=key: self.run(k)
                          ).pack(fill="x", pady=3)
        self._last_card = inner

    # --------------------------------------------------------------- actions
    def run(self, action: str) -> None:
        cols = self.picker.selected()
        n = len(cols)
        target = f"{n} column(s)" if n else "all columns"
        strategy = FILL_STRATEGIES[self.fill_strategy.get()]
        actions = {
            "fill": (f"Filled missing values ({strategy}) in {target}", lambda df: ops.fill_missing(df, strategy, cols)),
            "dropna": (f"Dropped rows with missing values in {target}", lambda df: ops.drop_missing_rows(df, cols)),
            "onehot": (f"One-hot encoded {', '.join(cols)}", lambda df: ops.one_hot_encode(df, cols)),
            "ordinal": (f"Ordinal encoded {', '.join(cols)}", lambda df: ops.ordinal_encode(df, cols)),
            "drop": (f"Dropped columns {', '.join(cols)}", lambda df: ops.drop_columns(df, cols)),
            "dupes": ("Removed duplicate rows", ops.drop_duplicates),
            "outliers": (f"Removed IQR outliers using {target}", lambda df: ops.remove_outliers_iqr(df, cols)),
            "scale": (f"Standardized {target}", lambda df: ops.standardize(df, cols)),
        }
        description, fn = actions[action]
        ds = self.app.dataset
        before = ds.df.shape
        try:
            ds.apply(description, fn)
        except Exception as exc:  # noqa: BLE001 — shown to the user
            self.app.notify(str(exc).strip("'\""), "error")
            return
        after = ds.df.shape
        delta = []
        if after[0] != before[0]:
            delta.append(f"{before[0] - after[0]:,} rows removed")
        if after[1] != before[1]:
            delta.append(f"{after[1] - before[1]:+} columns")
        self.app.notify(description + (f" ({', '.join(delta)})" if delta else "") + ".", "success")
        if action in ("drop", "onehot"):
            self.picker.select([])
        self.app.data_changed()

    def undo(self):
        ds = self.app.dataset
        if ds is None or not ds.can_undo:
            self.app.notify("Nothing to undo.", "warning")
            return
        step = ds.log[-1]
        ds.undo()
        self.app.notify(f"Undid: {step}", "info")
        self.app.data_changed()

    def reset(self):
        if self.app.dataset is None:
            return
        self.app.dataset.reset()
        self.app.notify("Dataset reset to the original file.", "info")
        self.app.data_changed()

    def refresh(self):
        ds = self.app.dataset
        self.picker.set_columns(ds.df)
        self.table.show(ds.df)
        self.history.configure(text="\n".join(f"{i}. {s}" for i, s in enumerate(ds.log, 1)) or "No steps yet.")
        self.undo_btn.configure(state="normal" if ds.can_undo else "disabled")

    def refresh_theme(self):
        self.table.refresh_theme()
