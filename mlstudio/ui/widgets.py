"""Reusable widgets: cards, metric tiles, data tables, plots, column pickers and a task runner."""
from __future__ import annotations

import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import ttk
from typing import Callable

import customtkinter as ctk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import theme


class Card(ctk.CTkFrame):
    """Rounded surface with an optional title and subtitle; put content in ``self.body``."""

    def __init__(self, master, title: str = "", subtitle: str = "", **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                         border_color=theme.BORDER, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        row = 0
        if title:
            ctk.CTkLabel(self, text=title, font=theme.font(15, "bold"), text_color=theme.TEXT,
                         anchor="w").grid(row=row, column=0, sticky="ew", padx=18, pady=(16, 0))
            row += 1
        if subtitle:
            ctk.CTkLabel(self, text=subtitle, font=theme.font(12), text_color=theme.MUTED, anchor="w",
                         justify="left", wraplength=520).grid(row=row, column=0, sticky="ew", padx=18, pady=(2, 0))
            row += 1
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=row, column=0, sticky="nsew", padx=18, pady=(10, 16))
        self.grid_rowconfigure(row, weight=1)


class MetricTile(ctk.CTkFrame):
    def __init__(self, master, label: str, value: str = "—", hint: str = ""):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=14, border_width=1, border_color=theme.BORDER)
        ctk.CTkLabel(self, text=label.upper(), font=theme.font(10, "bold"), text_color=theme.MUTED,
                     anchor="w").pack(fill="x", padx=16, pady=(12, 0))
        self.value = ctk.CTkLabel(self, text=value, font=theme.font(24, "bold"), text_color=theme.TEXT, anchor="w")
        self.value.pack(fill="x", padx=16)
        self.hint = ctk.CTkLabel(self, text=hint, font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self.hint.pack(fill="x", padx=16, pady=(0, 12))

    def set(self, value: str, hint: str | None = None) -> None:
        self.value.configure(text=value)
        if hint is not None:
            self.hint.configure(text=hint)


def tile_row(master, specs: list[tuple[str, str]]) -> list[MetricTile]:
    frame = ctk.CTkFrame(master, fg_color="transparent")
    tiles = []
    for i, (label, hint) in enumerate(specs):
        frame.grid_columnconfigure(i, weight=1, uniform="tiles")
        tile = MetricTile(frame, label, hint=hint)
        tile.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0 if i == len(specs) - 1 else 6))
        tiles.append(tile)
    frame.tiles = tiles
    return frame


class DataTable(ctk.CTkFrame):
    """Scrollable ttk.Treeview showing a DataFrame (rows capped for responsiveness)."""

    def __init__(self, master, max_rows: int = 500, height: int = 12):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=10, border_width=1,
                         border_color=theme.BORDER)
        self.max_rows = max_rows
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, show="headings", style="Studio.Treeview", height=height)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview, style="Studio.Vertical.TScrollbar")
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview, style="Studio.Horizontal.TScrollbar")
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        ys.grid(row=0, column=1, sticky="ns", pady=6)
        xs.grid(row=1, column=0, sticky="ew", padx=6)
        self.caption = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self.caption.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        self._stripe()

    def _stripe(self):
        self.tree.tag_configure("odd", background=theme.pick(theme.SURFACE_ALT))
        self.tree.tag_configure("even", background=theme.pick(theme.SURFACE))

    def refresh_theme(self):
        self._stripe()

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()
        self.caption.configure(text="")

    def show(self, df: pd.DataFrame, caption: str | None = None, float_format: str = "{:,.4g}") -> None:
        self.tree.delete(*self.tree.get_children())
        columns = [str(c) for c in df.columns]
        self.tree["columns"] = columns
        shown = df.head(self.max_rows)
        for col, src in zip(columns, df.columns):
            sample = [len(col)] + [len(str(v)) for v in shown[src].head(50)]
            width = int(min(max(max(sample) * 8 + 24, 80), 320))
            self.tree.heading(col, text=col, anchor="w")
            self.tree.column(col, width=width, minwidth=60, anchor="w", stretch=True)
        for i, row in enumerate(shown.itertuples(index=False)):
            values = [self._fmt(v, float_format) for v in row]
            self.tree.insert("", "end", values=values, tags=("odd" if i % 2 else "even",))
        if caption is None:
            caption = (f"Showing first {len(shown):,} of {len(df):,} rows" if len(df) > len(shown)
                       else f"{len(df):,} rows × {len(df.columns)} columns")
        self.caption.configure(text=caption)

    @staticmethod
    def _fmt(value, float_format):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        if isinstance(value, (float, np.floating)):
            return float_format.format(value)
        return str(value)


class StudioToolbar(NavigationToolbar2Tk):
    """Matplotlib toolbar without Back/Forward (their disabled icons render poorly on dark backgrounds)."""

    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] not in ("Back", "Forward")]


class PlotPanel(ctk.CTkFrame):
    """Embeds a Matplotlib figure. ``show(builder)`` calls ``builder(fig)`` and keeps it for theme redraws."""

    def __init__(self, master, placeholder: str = "Nothing to show yet.", toolbar: bool = True):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=10, border_width=1, border_color=theme.BORDER)
        self._builder: Callable[[Figure], None] | None = None
        self._figure_canvas = None
        self._toolbar = None
        self._with_toolbar = toolbar
        self.placeholder = ctk.CTkLabel(self, text=placeholder, font=theme.font(13), text_color=theme.MUTED)
        self.placeholder.pack(expand=True, fill="both", padx=20, pady=40)

    def show(self, builder: Callable[[Figure], None]) -> None:
        self._builder = builder
        self._render()

    def clear(self, placeholder: str | None = None) -> None:
        self._builder = None
        self._destroy_canvas()
        if placeholder:
            self.placeholder.configure(text=placeholder)
        self.placeholder.pack(expand=True, fill="both", padx=20, pady=40)

    def refresh_theme(self) -> None:
        if self._builder:
            self._render()

    def _destroy_canvas(self):
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._figure_canvas is not None:
            self._figure_canvas.get_tk_widget().destroy()
            self._figure_canvas = None

    def _render(self):
        theme.apply_mpl_style()
        self._destroy_canvas()
        self.placeholder.pack_forget()
        fig = Figure(figsize=(7, 4.6), dpi=100, layout="constrained")
        try:
            self._builder(fig)
        except Exception as exc:  # show plotting problems inside the panel rather than crashing
            fig.clear()
            ax = fig.add_subplot()
            ax.axis("off")
            ax.text(0.5, 0.5, f"Could not draw this chart:\n{exc}", ha="center", va="center", wrap=True)
        self._figure_canvas = FigureCanvasTkAgg(fig, master=self)
        if self._with_toolbar:
            self._toolbar = StudioToolbar(self._figure_canvas, self, pack_toolbar=False)
            self._toolbar.configure(background=theme.pick(theme.SURFACE))
            for child in self._toolbar.winfo_children():
                try:
                    child.configure(background=theme.pick(theme.SURFACE), activebackground=theme.pick(theme.SURFACE_ALT))
                except tk.TclError:
                    pass
                if hasattr(self._toolbar, "_set_image_for_button") and isinstance(child, tk.Button):
                    try:  # re-pick light/dark icon variants for the new background
                        self._toolbar._set_image_for_button(child)
                    except Exception:  # private Matplotlib API; icons are cosmetic
                        pass
            self._toolbar.update()
            self._toolbar.pack(side="bottom", fill="x", padx=8, pady=(0, 6))
        self._figure_canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=8, pady=8)
        self._figure_canvas.draw_idle()

    def save(self, path: str) -> None:
        if self._figure_canvas is not None:
            self._figure_canvas.figure.savefig(path, dpi=150, bbox_inches="tight")


class ColumnPicker(ctk.CTkFrame):
    """Searchable multi-select list of columns."""

    def __init__(self, master, height: int = 260, on_change: Callable[[], None] | None = None):
        super().__init__(master, fg_color="transparent")
        self._vars: dict[str, tk.BooleanVar] = {}
        self._boxes: dict[str, ctk.CTkCheckBox] = {}
        self._kinds: dict[str, str] = {}
        self._on_change = on_change
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")
        self.search = ctk.CTkEntry(top, placeholder_text="Filter columns…", height=32)
        self.search.pack(side="left", fill="x", expand=True)
        self.search.bind("<KeyRelease>", lambda _e: self._filter())
        for text, value in (("All", True), ("None", False)):
            ctk.CTkButton(top, text=text, width=48, height=32, fg_color=theme.SURFACE_ALT, text_color=theme.TEXT,
                          hover_color=theme.ACCENT_SOFT, command=lambda v=value: self.select_all(v)
                          ).pack(side="left", padx=(6, 0))
        self.list = ctk.CTkScrollableFrame(self, height=height, fg_color=theme.SURFACE_ALT, corner_radius=10)
        self.list.pack(fill="both", expand=True, pady=(8, 0))
        self.count = ctk.CTkLabel(self, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        self.count.pack(fill="x", pady=(4, 0))

    def set_columns(self, df: pd.DataFrame) -> None:
        previous = {c for c, v in self._vars.items() if v.get()}
        for box in self._boxes.values():
            box.destroy()
        self._vars.clear()
        self._boxes.clear()
        for col in df.columns:
            kind = "num" if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]) else "text"
            missing = int(df[col].isna().sum())
            label = f"{col}   ·  {kind}" + (f"  ·  {missing} missing" if missing else "")
            var = tk.BooleanVar(value=col in previous)
            box = ctk.CTkCheckBox(self.list, text=label, variable=var, font=theme.font(12), checkbox_width=18,
                                  checkbox_height=18, command=self._changed)
            self._vars[col], self._boxes[col], self._kinds[col] = var, box, kind
        self._filter()
        self._changed()

    def _filter(self):
        needle = self.search.get().lower().strip()
        for col, box in self._boxes.items():
            box.pack_forget()
            if needle in col.lower():
                box.pack(fill="x", anchor="w", padx=6, pady=3)

    def _changed(self):
        self.count.configure(text=f"{len(self.selected())} of {len(self._vars)} selected")
        if self._on_change:
            self._on_change()

    def select_all(self, value: bool = True) -> None:
        needle = self.search.get().lower().strip()
        for col, var in self._vars.items():
            if needle in col.lower():
                var.set(value)
        self._changed()

    def select(self, columns) -> None:
        for col, var in self._vars.items():
            var.set(col in columns)
        self._changed()

    def selected(self) -> list[str]:
        return [c for c, v in self._vars.items() if v.get()]


class TaskRunner:
    """Run slow work on a background thread and deliver the result on the Tk thread."""

    def __init__(self, widget, on_busy: Callable[[bool, str], None] | None = None):
        self.widget = widget
        self.on_busy = on_busy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlstudio")
        self._active = 0
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._active > 0

    def run(self, message: str, fn: Callable, *args, on_done: Callable | None = None,
            on_error: Callable[[Exception], None] | None = None) -> Future:
        self._active += 1
        if self.on_busy:
            self.on_busy(True, message)
        future = self._executor.submit(fn, *args)
        self._poll(future, on_done, on_error)
        return future

    def _poll(self, future: Future, on_done, on_error):
        if self._closed:
            return
        if not future.done():
            self.widget.after(60, lambda: self._poll(future, on_done, on_error))
            return
        self._active -= 1
        if self.on_busy:
            self.on_busy(self.busy, "")
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 — reported to the user
            if on_error:
                on_error(exc)
            return
        if on_done:
            on_done(result)

    def shutdown(self):
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
