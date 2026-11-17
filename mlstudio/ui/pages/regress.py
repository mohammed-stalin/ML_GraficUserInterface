"""Regression benchmark: compare models predicting a numeric column."""
from __future__ import annotations

import customtkinter as ctk
import numpy as np

from ...core import regression
from ...core.dataset import numeric_columns
from .. import theme
from ..widgets import DataTable, PlotPanel, tile_row
from .base import Page, control_label

LOG_OPTIONS = {"Auto (log if skewed)": None, "Always": True, "Never": False}


class RegressionPage(Page):
    title = "Regression"
    subtitle = "Predict a number. Six models are cross-validated and the best one is evaluated on held-out rows."

    def build(self, body):
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                                border_color=theme.BORDER, width=290)
        controls.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 14))
        controls.grid_propagate(False)
        c = ctk.CTkFrame(controls, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=16, pady=16)
        control_label(c, "TARGET COLUMN (numeric)").pack(fill="x")
        self.target = ctk.CTkOptionMenu(c, values=["—"], height=32, dynamic_resizing=False)
        self.target.pack(fill="x", pady=(4, 12))
        control_label(c, "LOG-SCALE TARGET").pack(fill="x")
        self.log = ctk.CTkOptionMenu(c, values=list(LOG_OPTIONS), height=32, dynamic_resizing=False)
        self.log.pack(fill="x", pady=(4, 4))
        ctk.CTkLabel(c, text="Modelling log(target) helps with skewed values such as prices.", font=theme.font(11),
                     text_color=theme.MUTED, anchor="w", justify="left", wraplength=250).pack(fill="x", pady=(0, 16))
        self.run_btn = ctk.CTkButton(c, text="Compare models", height=38, font=theme.font(13, "bold"),
                                     command=self.run)
        self.run_btn.pack(fill="x")
        self.messages = ctk.CTkLabel(c, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w",
                                     justify="left", wraplength=250)
        self.messages.pack(fill="x", pady=(16, 0))

        self.tiles = tile_row(body, [("Best model", "lowest CV error"), ("Test MAE", "mean absolute error"),
                                     ("Test R²", "1 = perfect"), ("vs baseline", "MAE reduction")])
        self.tiles.grid(row=0, column=1, sticky="ew", pady=(0, 14))
        self.tabs = ctk.CTkTabview(body, fg_color=theme.SURFACE, segmented_button_selected_color=theme.ACCENT,
                                   corner_radius=14, border_width=1, border_color=theme.BORDER, anchor="w")
        self.tabs.grid(row=1, column=1, sticky="nsew")
        tab = self.tabs.add("Leaderboard")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.table = DataTable(tab, height=8)
        self.table.grid(row=0, column=0, sticky="nsew")
        self.plots = {}
        for name in ("Predicted vs actual", "Residuals"):
            tab = self.tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            self.plots[name] = PlotPanel(tab, "Compare models to see results.")
            self.plots[name].grid(row=0, column=0, sticky="nsew")

    def refresh(self):
        cols = numeric_columns(self.df) or ["—"]
        current = self.target.get()
        self.target.configure(values=cols)
        if current not in cols:
            preferred = [c for c in cols if "price" in c.lower() or "target" in c.lower()]
            self.target.set(preferred[0] if preferred else cols[-1])

    def refresh_theme(self):
        self.table.refresh_theme()
        for p in self.plots.values():
            p.refresh_theme()

    def run(self):
        self.run_btn.configure(state="disabled")
        self.app.tasks.run("Cross-validating regression models…", regression.compare_regressors, self.df.copy(),
                           self.target.get(), LOG_OPTIONS[self.log.get()],
                           on_done=self._done, on_error=self._error)

    def _error(self, exc):
        self.run_btn.configure(state="normal")
        self.app.notify(str(exc), "error")

    def _done(self, r: regression.RegressionReport):
        self.run_btn.configure(state="normal")
        self.app.results["regression"] = r
        best = r.results.iloc[0]
        baseline = r.results.loc[r.results["Model"] == "Median baseline", "Test MAE"]
        best_t, mae_t, r2_t, gain_t = self.tiles.tiles
        best_t.set(best["Model"], "log-scale target" if r.log_target else "original scale")
        mae_t.set(f"{best['Test MAE']:,.4g}", f"RMSE {best['Test RMSE']:,.4g}")
        r2_t.set(f"{best['Test R2']:.3f}")
        if len(baseline):
            gain_t.set(f"{1 - best['Test MAE'] / baseline.iloc[0]:.0%}", "lower error than predicting the median")
        shown = r.results.copy()
        for col in ("CV MAE", "CV MAE std", "Test MAE", "Test RMSE"):
            shown[col] = shown[col].map(lambda v: f"{v:,.4g}")
        shown["Test R2"] = shown["Test R2"].map(lambda v: f"{v:.3f}")
        shown["Time (s)"] = shown["Time (s)"].map(lambda v: f"{v:.1f}")
        self.table.show(shown, caption=f"{r.train_rows:,} training / {r.test_rows:,} test rows · "
                                       f"{r.duplicates_removed:,} duplicates removed · ranked by 3-fold CV MAE")
        info = list(r.warnings)
        self.messages.configure(text="\n".join(info), text_color=theme.WARNING if info else theme.MUTED)
        self.plots["Predicted vs actual"].show(lambda fig: self._pred(fig, r))
        self.plots["Residuals"].show(lambda fig: self._resid(fig, r))
        self.app.notify(f"Best regression model: {best['Model']} (R² {best['Test R2']:.3f}).", "success")

    @staticmethod
    def _pred(fig, r):
        ax = fig.add_subplot()
        y, p = r.y_test.to_numpy(), r.y_pred
        ax.scatter(y, p, s=14, alpha=0.6, color=theme.SERIES[0], edgecolors="none")
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=theme.pick(theme.MUTED), linewidth=1, label="perfect")
        if r.log_target and lo > 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"Actual {r.target}")
        ax.set_ylabel(f"Predicted {r.target}")
        ax.set_title(f"{r.best_model}: predicted vs actual (test set)")
        ax.legend(loc="upper left")

    @staticmethod
    def _resid(fig, r):
        ax1, ax2 = fig.subplots(1, 2)
        resid = r.y_test.to_numpy() - r.y_pred
        ax1.scatter(r.y_pred, resid, s=12, alpha=0.6, color=theme.SERIES[0], edgecolors="none")
        ax1.axhline(0, color=theme.pick(theme.MUTED), linestyle="--", linewidth=1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual − predicted")
        ax1.set_title("Residuals vs prediction")
        clipped = np.clip(resid, *np.percentile(resid, [1, 99]))
        ax2.hist(clipped, bins=40, color=theme.SERIES[2])
        ax2.set_title("Residual distribution (1–99th pct)")
