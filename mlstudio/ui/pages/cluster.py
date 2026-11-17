"""K-Means clustering with an automatic choice of k."""
from __future__ import annotations

import customtkinter as ctk

from ...core import clustering
from .. import theme
from ..widgets import DataTable, PlotPanel, tile_row
from .base import Page, control_label, secondary_button

NONE = "(none)"


class ClusterPage(Page):
    title = "Cluster"
    subtitle = "Group similar rows with K-Means. Features are scaled automatically."

    def build(self, body):
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                                border_color=theme.BORDER, width=290)
        controls.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 14))
        controls.grid_propagate(False)
        c = ctk.CTkFrame(controls, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=16, pady=16)

        self.k_label = control_label(c, "CLUSTERS (k): 3")
        self.k_label.pack(fill="x")
        self.k = ctk.CTkSlider(c, from_=2, to=12, number_of_steps=10,
                               command=lambda v: self.k_label.configure(text=f"CLUSTERS (k): {int(v)}"))
        self.k.set(3)
        self.k.pack(fill="x", pady=(6, 14))
        control_label(c, "EXCLUDE COLUMN (e.g. a label)").pack(fill="x")
        self.exclude = ctk.CTkOptionMenu(c, values=[NONE], height=32, dynamic_resizing=False)
        self.exclude.pack(fill="x", pady=(4, 12))
        self.use_text = ctk.CTkSwitch(c, text="Include text columns (one-hot)", font=theme.font(12))
        self.use_text.pack(fill="x", pady=(0, 16))
        self.sweep_btn = secondary_button(c, "Find best k", self.sweep, height=38)
        self.sweep_btn.pack(fill="x")
        self.run_btn = ctk.CTkButton(c, text="Run K-Means", height=38, font=theme.font(13, "bold"), command=self.run)
        self.run_btn.pack(fill="x", pady=(8, 0))
        self.save_btn = secondary_button(c, "Add cluster labels to dataset", self.save_labels, height=34,
                                         state="disabled")
        self.save_btn.pack(fill="x", pady=(16, 0))
        self.info = ctk.CTkLabel(c, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w", justify="left",
                                 wraplength=250)
        self.info.pack(fill="x", pady=(12, 0))

        self.tiles = tile_row(body, [("Clusters", ""), ("Silhouette", "-1 to 1, higher is better"),
                                     ("Features used", ""), ("Largest cluster", "")])
        self.tiles.grid(row=0, column=1, sticky="ew", pady=(0, 14))
        self.tabs = ctk.CTkTabview(body, fg_color=theme.SURFACE, segmented_button_selected_color=theme.ACCENT,
                                   corner_radius=14, border_width=1, border_color=theme.BORDER, anchor="w")
        self.tabs.grid(row=1, column=1, sticky="nsew")
        self.plots = {}
        for name in ("Clusters", "Choosing k"):
            tab = self.tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            self.plots[name] = PlotPanel(tab, "Run K-Means to see clusters." if name == "Clusters"
                                         else "Click “Find best k” to compare cluster counts.")
            self.plots[name].grid(row=0, column=0, sticky="nsew")
        tab = self.tabs.add("Profiles")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.profile = DataTable(tab, height=14)
        self.profile.grid(row=0, column=0, sticky="nsew")

    def refresh(self):
        values = [NONE] + self.df.columns.tolist()
        current = self.exclude.get()
        self.exclude.configure(values=values)
        self.exclude.set(current if current in values else NONE)

    def refresh_theme(self):
        for p in self.plots.values():
            p.refresh_theme()
        self.profile.refresh_theme()

    def _exclude(self):
        value = self.exclude.get()
        return [] if value == NONE else [value]

    def _busy(self, busy):
        for b in (self.run_btn, self.sweep_btn):
            b.configure(state="disabled" if busy else "normal")

    def _error(self, exc):
        self._busy(False)
        self.app.notify(str(exc), "error")

    # --------------------------------------------------------------- actions
    def sweep(self):
        self._busy(True)
        self.app.tasks.run("Comparing k = 2…10…", clustering.sweep_k, self.df.copy(), 2, 10, self._exclude(),
                           bool(self.use_text.get()), on_done=self._swept, on_error=self._error)

    def _swept(self, table):
        self._busy(False)
        best = clustering.best_k(table)
        self.k.set(best)
        self.k_label.configure(text=f"CLUSTERS (k): {best}")
        self.plots["Choosing k"].show(lambda fig: self._sweep_chart(fig, table, best))
        self.tabs.set("Choosing k")
        self.app.notify(f"Best silhouette at k = {best}. The slider was set; click “Run K-Means”.", "success")

    def run(self):
        self._busy(True)
        k = int(self.k.get())
        self.app.tasks.run(f"Running K-Means with k = {k}…", clustering.run_kmeans, self.df.copy(), k,
                           self._exclude(), bool(self.use_text.get()), on_done=self._done, on_error=self._error)

    def _done(self, r: clustering.ClusterResult):
        self._busy(False)
        self.app.results["clustering"] = r
        self.save_btn.configure(state="normal")
        k, sil, feats, largest = self.tiles.tiles
        k.set(str(r.k))
        sil.set("—" if r.silhouette is None else f"{r.silhouette:.3f}")
        feats.set(str(len(r.spec.used)), f"{len(r.spec.numeric)} numeric · {len(r.spec.categorical)} text")
        largest.set(f"{r.sizes.max() / r.sizes.sum():.0%}", f"{r.sizes.max():,} rows")
        dropped = ", ".join(f"{c} ({why})" for c, why in r.spec.dropped.items())
        self.info.configure(text=f"Ignored: {dropped}" if dropped else "")
        self.plots["Clusters"].show(lambda fig: self._scatter(fig, r))
        self.profile.show(r.profile.round(2), caption="Mean of each numeric feature per cluster")
        self.tabs.set("Clusters")
        self.app.notify(f"K-Means found {r.k} clusters.", "success")

    def save_labels(self):
        r = self.app.results.get("clustering")
        ds = self.app.dataset
        if r is None or len(r.labels) != len(ds.df):
            self.app.notify("Run K-Means again: the dataset changed since the last run.", "warning")
            return
        labels = r.labels

        def add(df):
            df["Cluster"] = labels
            return df
        ds.apply(f"Added K-Means cluster labels (k = {r.k})", add)
        self.app.notify("Added a 'Cluster' column to the dataset.", "success")
        self.app.data_changed()

    # ------------------------------------------------------------------ plots
    @staticmethod
    def _scatter(fig, r):
        ax = fig.add_subplot()
        for i in range(r.k):
            mask = r.labels == i
            ax.scatter(r.coords[mask, 0], r.coords[mask, 1], s=14, alpha=0.75, edgecolors="none",
                       color=theme.SERIES[i % len(theme.SERIES)], label=f"Cluster {i} ({mask.sum():,})")
        var = r.explained_variance
        ax.set_xlabel(f"PC1 ({var[0]:.0%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.0%} variance)" if len(var) > 1 else "PC2")
        ax.set_title("K-Means clusters (PCA projection of scaled features)")
        ax.legend(loc="best")

    @staticmethod
    def _sweep_chart(fig, table, best):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(table["k"], table["inertia"], marker="o", color=theme.SERIES[0])
        ax1.set_title("Elbow: within-cluster distance")
        ax1.set_xlabel("k")
        ax2.plot(table["k"], table["silhouette"], marker="o", color=theme.SERIES[2])
        ax2.axvline(best, color=theme.SERIES[1], linestyle="--", linewidth=1)
        ax2.set_title(f"Silhouette (best k = {best})")
        ax2.set_xlabel("k")
