"""Colours, fonts and theme helpers for CustomTkinter, ttk and Matplotlib."""
from __future__ import annotations

import sys
from tkinter import ttk

import customtkinter as ctk
import matplotlib as mpl

# (light, dark) pairs, the format CustomTkinter expects
BG = ("#F4F6FB", "#0F141C")
SIDEBAR = ("#FFFFFF", "#151B26")
SURFACE = ("#FFFFFF", "#1B2230")
SURFACE_ALT = ("#EEF2F8", "#232C3D")
BORDER = ("#E2E8F0", "#2A3446")
TEXT = ("#0F172A", "#E6EAF2")
MUTED = ("#64748B", "#8A94A8")
ACCENT = ("#2563EB", "#4F8BFF")
ACCENT_HOVER = ("#1D4ED8", "#3B74E6")
ACCENT_SOFT = ("#DBE7FF", "#1E2D4D")
SUCCESS = ("#15803D", "#4ADE80")
WARNING = ("#B45309", "#FBBF24")
DANGER = ("#B91C1C", "#F87171")

# Categorical palette for charts (colour-blind friendly ordering)
SERIES = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16",
          "#64748B", "#F97316"]

FAMILY = "Segoe UI" if sys.platform == "win32" else ("SF Pro Text" if sys.platform == "darwin" else "DejaVu Sans")


def font(size: int = 13, weight: str = "normal") -> tuple:
    return (FAMILY, size, weight)


def is_dark() -> bool:
    return ctk.get_appearance_mode() == "Dark"


def pick(pair: tuple[str, str]) -> str:
    return pair[1] if is_dark() else pair[0]


def apply_ttk_style(root) -> None:
    """Make ttk.Treeview match the CustomTkinter palette."""
    style = ttk.Style(root)
    style.theme_use("clam")
    surface, alt, text, muted, border = map(pick, (SURFACE, SURFACE_ALT, TEXT, MUTED, BORDER))
    style.configure("Studio.Treeview", background=surface, fieldbackground=surface, foreground=text,
                    rowheight=28, borderwidth=0, font=font(11))
    style.configure("Studio.Treeview.Heading", background=alt, foreground=muted, relief="flat",
                    borderwidth=0, font=font(11, "bold"), padding=(8, 6))
    style.map("Studio.Treeview", background=[("selected", pick(ACCENT))], foreground=[("selected", "#FFFFFF")])
    style.map("Studio.Treeview.Heading", background=[("active", pick(ACCENT_SOFT))])
    style.layout("Studio.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    style.configure("Studio.Vertical.TScrollbar", background=alt, troughcolor=surface, bordercolor=surface,
                    arrowcolor=muted, lightcolor=alt, darkcolor=alt)
    style.configure("Studio.Horizontal.TScrollbar", background=alt, troughcolor=surface, bordercolor=surface,
                    arrowcolor=muted, lightcolor=alt, darkcolor=alt)
    style.configure("TCombobox", fieldbackground=surface)
    root.option_add("*TCombobox*Listbox*Background", surface)


def apply_mpl_style() -> None:
    surface, text, muted, border = map(pick, (SURFACE, TEXT, MUTED, BORDER))
    mpl.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "axes.edgecolor": border, "axes.labelcolor": text, "axes.titlecolor": text,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": muted, "ytick.color": muted, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": text, "grid.color": border, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "legend.frameon": False, "legend.fontsize": 8,
        "font.family": "sans-serif",
    })
