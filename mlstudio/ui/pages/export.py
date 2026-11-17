"""Export a PDF report or the cleaned dataset."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ...core.report import build_report
from .. import theme
from ..widgets import Card
from .base import Page


class ExportPage(Page):
    title = "Export"
    subtitle = "Save your work: a PDF report or the prepared dataset."

    def build(self, body):
        body.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        report = Card(body, "PDF report", "Overview, column profile, statistics, preparation steps and the latest "
                                          "classification, regression and clustering results.")
        report.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.contents = ctk.CTkLabel(report.body, text="", font=theme.font(12), text_color=theme.MUTED,
                                     anchor="w", justify="left")
        self.contents.pack(fill="x", pady=(0, 12))
        ctk.CTkButton(report.body, text="Export PDF report…", height=38, font=theme.font(13, "bold"),
                      command=self.export_pdf).pack(anchor="w")

        data = Card(body, "Prepared dataset", "Save the dataset with every preparation step applied.")
        data.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.steps = ctk.CTkLabel(data.body, text="", font=theme.font(12), text_color=theme.MUTED, anchor="w",
                                  justify="left", wraplength=480)
        self.steps.pack(fill="x", pady=(0, 12))
        ctk.CTkButton(data.body, text="Save as CSV or Excel…", height=38, font=theme.font(13, "bold"),
                      command=self.export_data).pack(anchor="w")

    def on_show(self):
        self.dirty = True  # results may have changed without a data change
        super().on_show()

    def refresh(self):
        ds, res = self.app.dataset, self.app.results
        lines = [f"✓  Dataset overview ({len(ds.df):,} rows × {len(ds.df.columns)} columns)",
                 f"✓  {len(ds.log)} preparation step(s)"]
        for key, label in (("classification", "Classification"), ("regression", "Regression"),
                           ("clustering", "Clustering")):
            lines.append(("✓  " if res[key] is not None else "–  ") + label +
                         ("" if res[key] is not None else " (not run yet)"))
        self.contents.configure(text="\n".join(lines))
        self.steps.configure(text="\n".join(f"{i}. {s}" for i, s in enumerate(ds.log, 1)) or "No changes applied.")

    def export_pdf(self):
        ds = self.app.dataset
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
                                            initialfile=f"{Path(ds.name).stem}_report.pdf")
        if not path:
            return
        res = self.app.results
        self.app.tasks.run("Writing PDF report…", build_report, path, ds, res["classification"], res["regression"],
                           res["clustering"], on_done=lambda p: self.app.notify(f"Report saved to {p}", "success"),
                           on_error=lambda e: self.app.notify(f"Could not write the report: {e}", "error"))

    def export_data(self):
        ds = self.app.dataset
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"{Path(ds.name).stem}_prepared.csv",
                                            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if not path:
            return
        try:
            if path.lower().endswith(".xlsx"):
                ds.df.to_excel(path, index=False)
            else:
                ds.df.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            self.app.notify(f"Could not save: {exc}", "error")
            return
        self.app.notify(f"Dataset saved to {path}", "success")
