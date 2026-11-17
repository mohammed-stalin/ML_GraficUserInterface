"""Supervised classification: train one model or compare all of them."""
from __future__ import annotations

import customtkinter as ctk
import numpy as np
import pandas as pd

from ...core import classification as clf
from .. import theme
from ..widgets import DataTable, PlotPanel, tile_row
from .base import Page, control_label, secondary_button


class ClassifyPage(Page):
    title = "Classify"
    subtitle = "Predict a category. Preprocessing, scaling and encoding are handled automatically."

    def build(self, body):
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=14, border_width=1,
                                border_color=theme.BORDER, width=290)
        controls.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 14))
        controls.grid_propagate(False)
        c = ctk.CTkFrame(controls, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=16, pady=16)

        control_label(c, "TARGET COLUMN").pack(fill="x")
        self.target = ctk.CTkOptionMenu(c, values=["—"], height=32, dynamic_resizing=False,
                                        command=lambda _v: self._target_changed())
        self.target.pack(fill="x", pady=(4, 2))
        self.target_info = ctk.CTkLabel(c, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w",
                                        justify="left", wraplength=250)
        self.target_info.pack(fill="x", pady=(0, 10))
        control_label(c, "MODEL").pack(fill="x")
        self.model = ctk.CTkOptionMenu(c, values=list(clf.CLASSIFIERS), height=32, dynamic_resizing=False)
        self.model.set("Random Forest")
        self.model.pack(fill="x", pady=(4, 12))
        self.split_label = control_label(c, "TEST SET: 20%")
        self.split_label.pack(fill="x")
        self.split = ctk.CTkSlider(c, from_=10, to=40, number_of_steps=6,
                                   command=lambda v: self.split_label.configure(text=f"TEST SET: {int(v)}%"))
        self.split.set(20)
        self.split.pack(fill="x", pady=(6, 12))
        self.use_text = ctk.CTkSwitch(c, text="Use text columns (one-hot)", font=theme.font(12))
        self.use_text.select()
        self.use_text.pack(fill="x", pady=(0, 16))
        self.train_btn = ctk.CTkButton(c, text="Train model", height=38, font=theme.font(13, "bold"),
                                       command=self.train)
        self.train_btn.pack(fill="x")
        self.compare_btn = secondary_button(c, "Compare all models", self.compare, height=38)
        self.compare_btn.pack(fill="x", pady=(8, 0))
        self.messages = ctk.CTkLabel(c, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w",
                                     justify="left", wraplength=250)
        self.messages.pack(fill="x", pady=(16, 0))

        self.tiles = tile_row(body, [("Accuracy", "held-out test set"), ("Balanced accuracy", "mean recall per class"),
                                     ("Macro F1", "precision/recall balance"), ("CV accuracy", "k-fold on training set")])
        self.tiles.grid(row=0, column=1, sticky="ew", pady=(0, 14))

        self.tabs = ctk.CTkTabview(body, fg_color=theme.SURFACE, segmented_button_selected_color=theme.ACCENT,
                                   corner_radius=14, border_width=1, border_color=theme.BORDER, anchor="w",
                                   command=self._tab_changed)
        self.tabs.grid(row=1, column=1, sticky="nsew")
        self.plots, self.tables = {}, {}
        for name in ("Confusion matrix", "Report", "Feature importance", "Projection", "Decision tree", "Comparison"):
            tab = self.tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            if name in ("Report", "Comparison"):
                widget = DataTable(tab, height=14)
                self.tables[name] = widget
            else:
                widget = PlotPanel(tab, "Train a model to see results.")
                self.plots[name] = widget
            widget.grid(row=0, column=0, sticky="nsew")
        self.plots["Feature importance"].placeholder.configure(text="Open this tab after training to compute "
                                                                    "permutation importance.")
        self.plots["Comparison"] = PlotPanel(self.tabs.tab("Comparison"), "", toolbar=False)
        self.plots["Comparison"].grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.tabs.tab("Comparison").grid_rowconfigure(1, weight=1)
        self._importance_done = False

    # ----------------------------------------------------------------- state
    def refresh(self):
        df = self.df
        cols = df.columns.tolist()
        current = self.target.get()
        self.target.configure(values=cols)
        if current not in cols:
            likely = [c for c in cols if 2 <= df[c].nunique() <= 20]
            self.target.set(likely[-1] if likely else cols[-1])
        self._target_changed()

    def _target_changed(self):
        df, target = self.df, self.target.get()
        if df is None or target not in df:
            return
        counts = df[target].value_counts()
        n = len(counts)
        if n > 20 and pd.api.types.is_numeric_dtype(df[target]):
            text, color = f"{n} distinct numeric values — this looks like a regression target.", theme.WARNING
        else:
            top = ", ".join(f"{k} ({v})" for k, v in counts.head(4).items())
            text, color = f"{n} classes: {top}{' …' if n > 4 else ''}", theme.MUTED
        self.target_info.configure(text=text, text_color=color)

    def _busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.train_btn.configure(state=state)
        self.compare_btn.configure(state=state)

    def _error(self, exc):
        self._busy(False)
        self.app.notify(str(exc), "error")

    # --------------------------------------------------------------- actions
    def train(self):
        self._busy(True)
        df, target, model = self.df.copy(), self.target.get(), self.model.get()
        self.app.tasks.run(f"Training {model}…", clf.train_classifier, df, target, model, self.split.get() / 100,
                           bool(self.use_text.get()), on_done=self._trained, on_error=self._error)

    def compare(self):
        self._busy(True)
        df, target = self.df.copy(), self.target.get()
        self.app.tasks.run("Cross-validating all models…", clf.compare_classifiers, df, target, None,
                           self.split.get() / 100, bool(self.use_text.get()),
                           on_done=self._compared, on_error=self._error)

    def _trained(self, r: clf.ClassificationResult):
        self._busy(False)
        self.app.results["classification"] = r
        acc, bal, f1, cv = self.tiles.tiles
        acc.set(f"{r.accuracy:.1%}", f"{r.test_rows:,} test rows")
        bal.set(f"{r.balanced_accuracy:.1%}")
        f1.set(f"{r.f1_macro:.3f}")
        cv.set(f"{r.cv_accuracy:.1%}", f"± {r.cv_accuracy_std:.1%} across folds")
        self.messages.configure(text="\n".join(r.warnings), text_color=theme.WARNING if r.warnings else theme.MUTED)
        self.plots["Confusion matrix"].show(lambda fig: self._confusion(fig, r))
        self.tables["Report"].show(r.report.round(3), caption=f"{r.model_name} · target '{r.target}'")
        self.plots["Projection"].show(lambda fig: self._projection(fig, r))
        if r.model_name in clf.TREE_MODELS:
            self.plots["Decision tree"].show(lambda fig: self._tree(fig, r))
        else:
            self.plots["Decision tree"].clear("Select “Decision Tree (ID3)” to see the tree diagram.")
        self._importance_done = False
        self.plots["Feature importance"].clear("Open this tab to compute permutation importance.")
        if self.tabs.get() in ("Comparison", "Decision tree") and r.model_name not in clf.TREE_MODELS:
            self.tabs.set("Confusion matrix")
        self._tab_changed()
        self.app.notify(f"{r.model_name}: {r.accuracy:.1%} test accuracy, {r.cv_accuracy:.1%} CV accuracy.", "success")

    def _compared(self, table: pd.DataFrame):
        self._busy(False)
        shown = table.copy()
        for col in shown.columns[1:]:
            shown[col] = shown[col].map(lambda v: f"{v:.3f}")
        self.tables["Comparison"].show(shown, caption="Ranked by cross-validated macro F1 on the training split")
        self.plots["Comparison"].show(lambda fig: self._comparison_chart(fig, table))
        self.tabs.set("Comparison")
        self.app.notify(f"Best model: {table.iloc[0]['Model']} (CV macro F1 {table.iloc[0]['CV macro F1']:.3f}).",
                        "success")

    def _tab_changed(self):
        r = self.app.results.get("classification")
        if self.tabs.get() == "Feature importance" and r is not None and not self._importance_done:
            self._importance_done = True
            self.app.tasks.run("Computing permutation importance…", clf.feature_importance, r,
                               on_done=lambda imp: self.plots["Feature importance"].show(
                                   lambda fig: self._importance(fig, imp)),
                               on_error=self._error)

    def refresh_theme(self):
        for p in self.plots.values():
            p.refresh_theme()
        for t in self.tables.values():
            t.refresh_theme()

    # ------------------------------------------------------------------ plots
    @staticmethod
    def _confusion(fig, r):
        ax = fig.add_subplot()
        cm = r.confusion
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        labels = [str(label) for label in r.labels]
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        if len(labels) <= 15:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.5 else "#0F172A")
        ax.set_title(f"{r.model_name} — confusion matrix (row-normalised)")

    @staticmethod
    def _projection(fig, r):
        coords, var = clf.projection(r)
        ax = fig.add_subplot()
        y_pred = pd.Series(r.y_pred).astype(str).to_numpy()
        wrong = y_pred != r.y_test.astype(str).to_numpy()
        for i, label in enumerate(r.labels[:10]):
            mask = (y_pred == str(label)) & ~wrong
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.75, label=str(label),
                       color=theme.SERIES[i % len(theme.SERIES)], edgecolors="none")
        ax.scatter(coords[wrong, 0], coords[wrong, 1], s=40, facecolors="none", edgecolors=theme.pick(theme.DANGER),
                   linewidths=1.2, label=f"misclassified ({wrong.sum()})")
        ax.set_xlabel(f"PC1 ({var[0]:.0%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.0%} variance)" if len(var) > 1 else "PC2")
        ax.set_title("Test set predictions (PCA projection)")
        ax.legend(loc="best", markerscale=1.2)

    @staticmethod
    def _importance(fig, imp: pd.Series):
        ax = fig.add_subplot()
        imp = imp[::-1]
        colors = [theme.SERIES[0] if v >= 0 else theme.pick(theme.MUTED) for v in imp.values]
        ax.barh([str(i) for i in imp.index], imp.values, color=colors)
        ax.set_xlabel("Drop in accuracy when the column is shuffled")
        ax.set_title("Permutation importance (test set)")
        ax.grid(axis="y", visible=False)

    @staticmethod
    def _tree(fig, r):
        from sklearn.tree import plot_tree
        model = r.pipeline.named_steps["model"]
        ax = fig.add_subplot()
        ax.grid(False)
        plot_tree(model, max_depth=3, feature_names=r.feature_names, class_names=[str(c) for c in model.classes_],
                  filled=True, rounded=True, impurity=False, fontsize=7, ax=ax)
        ax.set_title(f"Decision tree (top 3 of {model.get_depth()} levels, {model.get_n_leaves()} leaves)")

    @staticmethod
    def _comparison_chart(fig, table):
        ax = fig.add_subplot()
        t = table[::-1]
        ax.barh(t["Model"], t["CV accuracy"], xerr=t["CV std"], color=theme.SERIES[0], alpha=0.85,
                label="CV accuracy", error_kw={"ecolor": theme.pick(theme.MUTED), "capsize": 3})
        ax.scatter(t["Test accuracy"], t["Model"], color=theme.SERIES[1], zorder=3, label="Test accuracy")
        low = max(0.0, min(t["CV accuracy"].min(), t["Test accuracy"].min()) - 0.1)
        ax.set_xlim(low, 1.0)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_title("Model comparison")
