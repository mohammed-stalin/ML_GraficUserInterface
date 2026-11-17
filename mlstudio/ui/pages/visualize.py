"""Interactive charts embedded in the window (no blocking pop-up windows)."""
from __future__ import annotations

import math

import customtkinter as ctk
import numpy as np
import pandas as pd
import seaborn as sns

from ...core.dataset import categorical_columns, numeric_columns
from .. import theme
from ..widgets import PlotPanel
from .base import Page, control_label, secondary_button

ALL_NUMERIC = "All numeric columns"
NONE = "(none)"
MAX_POINTS = 5000
MAX_GROUPS = 12

# chart -> which controls it uses: x, y, hue
CHARTS = {
    "Histogram": ("x", "hue"),
    "Box plot": ("y", "x"),
    "Scatter": ("x", "y", "hue"),
    "Category counts": ("x",),
    "Mean by category": ("x", "y"),
    "Correlation": (),
}
HELP = {
    "Histogram": "Distribution of one numeric column, or a grid of all numeric columns. Optional colour split.",
    "Box plot": "Median, quartiles and outliers. Choose a value column and optionally a grouping column.",
    "Scatter": "Relationship between two numeric columns (up to 5,000 sampled points).",
    "Category counts": "How often each category appears (top 20).",
    "Mean by category": "Average of a numeric column per category with 95% confidence intervals.",
    "Correlation": "Pearson correlation between numeric columns.",
}


class VisualizePage(Page):
    title = "Visualize"
    subtitle = "Pick a chart and the columns to plot. Charts update as you change settings."

    def build(self, body):
        secondary_button(self.actions, "Save chart as image…", self.save, height=36).pack()
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        controls = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                                border_color=theme.BORDER, width=290)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        controls.grid_propagate(False)
        inner = ctk.CTkFrame(controls, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=16)

        control_label(inner, "CHART").pack(fill="x")
        self.chart = ctk.CTkSegmentedButton(inner, values=list(CHARTS)[:3], command=lambda _v: self._chart_changed())
        self.chart.pack(fill="x", pady=(6, 4))
        self.chart2 = ctk.CTkSegmentedButton(inner, values=list(CHARTS)[3:], command=lambda _v: self._chart_changed(2))
        self.chart2.pack(fill="x")
        self.chart.set("Histogram")
        self.help = ctk.CTkLabel(inner, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w",
                                 justify="left", wraplength=250)
        self.help.pack(fill="x", pady=(8, 10))

        self.menus = {}
        for key, label in (("x", "X / COLUMN"), ("y", "Y / VALUE"), ("hue", "COLOUR BY")):
            frame = ctk.CTkFrame(inner, fg_color="transparent")
            control_label(frame, label).pack(fill="x")
            menu = ctk.CTkOptionMenu(frame, values=[NONE], height=32, dynamic_resizing=False,
                                     command=lambda _v: self.render())
            menu.pack(fill="x", pady=(4, 10))
            self.menus[key] = (frame, menu)
        self.note = ctk.CTkLabel(inner, text="", font=theme.font(11), text_color=theme.WARNING, anchor="w",
                                 justify="left", wraplength=250)
        self.note.pack(fill="x", side="bottom")

        self.plot = PlotPanel(body, "Choose a chart to begin.")
        self.plot.grid(row=0, column=1, sticky="nsew")

    def _chart_name(self) -> str:
        return self.chart.get() or self.chart2.get()

    def _chart_changed(self, source: int = 1):
        (self.chart2 if source == 1 else self.chart).set("")
        self._layout_controls()
        self.render()

    def _layout_controls(self):
        name = self._chart_name()
        used = CHARTS[name]
        self.help.configure(text=HELP[name])
        for key, (frame, _menu) in self.menus.items():
            frame.pack_forget()
        for key in ("x", "y", "hue"):
            if key in used:
                self.menus[key][0].pack(fill="x", before=self.note)
        df = self.df
        num, cat = numeric_columns(df), categorical_columns(df)
        groupable = [c for c in df.columns if df[c].nunique() <= 30]
        options = {
            "Histogram": {"x": [ALL_NUMERIC] + num, "hue": [NONE] + groupable},
            "Box plot": {"y": [ALL_NUMERIC] + num, "x": [NONE] + groupable},
            "Scatter": {"x": num, "y": num, "hue": [NONE] + df.columns.tolist()},
            "Category counts": {"x": cat + [c for c in num if c in groupable] or df.columns.tolist()},
            "Mean by category": {"x": groupable or df.columns.tolist(), "y": num},
            "Correlation": {},
        }[name]
        for key, values in options.items():
            menu = self.menus[key][1]
            values = values or [NONE]
            current = menu.get()
            menu.configure(values=values)
            if current not in values:
                default = values[1] if key == "y" and name == "Scatter" and len(values) > 1 else values[0]
                menu.set(default)

    def refresh(self):
        self._layout_controls()
        self.render()

    def refresh_theme(self):
        self.plot.refresh_theme()

    # ----------------------------------------------------------------- render
    def render(self):
        df = self.df
        if df is None:
            return
        name = self._chart_name()
        x, y, hue = (self.menus[k][1].get() for k in ("x", "y", "hue"))
        hue = None if hue == NONE else hue
        self.note.configure(text="")
        if name == "Scatter" and len(df) > MAX_POINTS:
            self.note.configure(text=f"Showing a random sample of {MAX_POINTS:,} rows.")
        drawers = {
            "Histogram": lambda fig: self._hist(fig, df, x, hue),
            "Box plot": lambda fig: self._box(fig, df, y, None if x == NONE else x),
            "Scatter": lambda fig: self._scatter(fig, df, x, y, hue),
            "Category counts": lambda fig: self._counts(fig, df, x),
            "Mean by category": lambda fig: self._means(fig, df, x, y),
            "Correlation": lambda fig: self._corr(fig, df),
        }
        self.plot.show(drawers[name])

    @staticmethod
    def _grid(fig, n):
        cols = min(4, n)
        rows = math.ceil(n / cols)
        return [fig.add_subplot(rows, cols, i + 1) for i in range(n)]

    def _hist(self, fig, df, x, hue):
        cols = numeric_columns(df) if x == ALL_NUMERIC else [x]
        if not cols or cols == [NONE]:
            raise ValueError("No numeric columns to plot.")
        cols = cols[:16]
        kde = len(df) <= 20000
        hue_ok = hue and df[hue].nunique() <= MAX_GROUPS
        for ax, col in zip(self._grid(fig, len(cols)), cols):
            sns.histplot(data=df, x=col, hue=hue if hue_ok else None, bins="auto", kde=kde, ax=ax,
                         palette=theme.SERIES[:df[hue].nunique()] if hue_ok else None,
                         color=theme.SERIES[0], element="step" if hue_ok else "bars", legend=len(cols) == 1)
            ax.set_title(col)
            ax.set_xlabel("")

    def _box(self, fig, df, y, by):
        cols = numeric_columns(df) if y == ALL_NUMERIC else [y]
        if not cols or cols == [NONE]:
            raise ValueError("No numeric columns to plot.")
        cols = cols[:16]
        for ax, col in zip(self._grid(fig, len(cols)), cols):
            if by:
                order = df[by].value_counts().index[:MAX_GROUPS]
                sns.boxplot(data=df[df[by].isin(order)], x=by, y=col, order=order, ax=ax, color=theme.SERIES[0],
                            fliersize=2, linewidth=1)
                ax.tick_params(axis="x", rotation=30)
            else:
                sns.boxplot(data=df, y=col, ax=ax, color=theme.SERIES[0], fliersize=2, linewidth=1, width=0.4)
            ax.set_title(col)

    def _scatter(self, fig, df, x, y, hue):
        if x == NONE or y == NONE:
            raise ValueError("Scatter plots need two numeric columns.")
        data = df.sample(MAX_POINTS, random_state=0) if len(df) > MAX_POINTS else df
        ax = fig.add_subplot()
        if hue and data[hue].nunique() > MAX_GROUPS and pd.api.types.is_numeric_dtype(data[hue]):
            points = ax.scatter(data[x], data[y], c=data[hue], cmap="viridis", s=14, alpha=0.8)
            fig.colorbar(points, ax=ax, label=hue)
        elif hue:
            top = data[hue].value_counts().index[:MAX_GROUPS]
            data = data[data[hue].isin(top)]
            sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, s=18, alpha=0.8, edgecolor="none",
                            palette=theme.SERIES[:len(top)])
        else:
            ax.scatter(data[x], data[y], s=14, alpha=0.7, color=theme.SERIES[0], edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")

    def _counts(self, fig, df, x):
        counts = df[x].astype(str).value_counts().head(20)
        ax = fig.add_subplot()
        ax.barh(counts.index[::-1], counts.values[::-1], color=theme.SERIES[0])
        ax.set_title(f"{x}: most frequent values")
        ax.set_xlabel("Rows")
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(counts.values[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=theme.pick(theme.MUTED))

    def _means(self, fig, df, x, y):
        if y == NONE:
            raise ValueError("Choose a numeric value column.")
        top = df[x].value_counts().index[:20]
        data = df[df[x].isin(top)]
        order = data.groupby(x)[y].mean().sort_values().index
        ax = fig.add_subplot()
        sns.pointplot(data=data, x=y, y=x, order=order, ax=ax, color=theme.SERIES[0], linestyle="none",
                      errorbar=("ci", 95), markersize=5)
        ax.set_title(f"Mean {y} by {x}")

    def _corr(self, fig, df):
        num = df[numeric_columns(df)]
        if num.shape[1] < 2:
            raise ValueError("Correlation needs at least two numeric columns.")
        num = num.loc[:, num.nunique() > 1].iloc[:, :30]
        corr = num.corr()
        ax = fig.add_subplot()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=len(corr) <= 12, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    center=0, square=True, linewidths=0.5, linecolor=theme.pick(theme.SURFACE),
                    cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"fontsize": 8})
        ax.grid(False)
        ax.set_title("Correlation matrix (Pearson)")

    def save(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("SVG", "*.svg"),
                                                                               ("PDF", "*.pdf")])
        if path:
            self.plot.save(path)
            self.app.notify(f"Chart saved to {path}", "success")
