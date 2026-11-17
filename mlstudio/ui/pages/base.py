"""Base class for application pages."""
from __future__ import annotations

import customtkinter as ctk

from .. import theme


class Page(ctk.CTkFrame):
    """A full-height view. Subclasses build widgets in ``build`` and update them in ``refresh``.

    ``refresh`` runs when the page is shown after the dataset changed, so pages never
    show stale columns. Pages that need data show an empty state until one is loaded.
    """

    title = ""
    subtitle = ""
    needs_data = True

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.dirty = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 12))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=self.title, font=theme.font(24, "bold"), text_color=theme.TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text=self.subtitle, font=theme.font(13), text_color=theme.MUTED,
                     anchor="w").grid(row=1, column=0, sticky="w")
        self.actions = ctk.CTkFrame(header, fg_color="transparent", width=1, height=1)
        self.actions.grid(row=0, column=1, rowspan=2, sticky="e")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 16))
        self.empty = self._build_empty_state()
        self.build(self.body)

    # ----------------------------------------------------------------- hooks
    def build(self, body: ctk.CTkFrame) -> None:
        """Create widgets once."""

    def refresh(self) -> None:
        """Update widgets from ``self.app.dataset``."""

    def refresh_theme(self) -> None:
        """Redraw anything that does not follow CustomTkinter colours (plots, tables)."""

    # -------------------------------------------------------------- plumbing
    @property
    def df(self):
        return self.app.dataset.df if self.app.dataset is not None else None

    def on_show(self) -> None:
        has_data = self.app.dataset is not None
        if self.needs_data and not has_data:
            self.body.grid_remove()
            self.empty.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 16))
            return
        self.empty.grid_remove()
        self.body.grid()
        if self.dirty:
            self.dirty = False
            self.refresh()

    def _build_empty_state(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=16, border_width=1, border_color=theme.BORDER)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.45, anchor="center")
        ctk.CTkLabel(inner, text="No dataset loaded", font=theme.font(20, "bold"), text_color=theme.TEXT).pack()
        ctk.CTkLabel(inner, text="Open a CSV or Excel file to use this page.", font=theme.font(13),
                     text_color=theme.MUTED).pack(pady=(4, 16))
        ctk.CTkButton(inner, text="Open dataset…", height=38, font=theme.font(13, "bold"),
                      command=self.app.open_dataset).pack()
        return frame


def control_label(master, text: str):
    return ctk.CTkLabel(master, text=text, font=theme.font(12, "bold"), text_color=theme.MUTED, anchor="w")


def secondary_button(master, text: str, command, **kwargs):
    return ctk.CTkButton(master, text=text, command=command, fg_color=theme.SURFACE_ALT, text_color=theme.TEXT,
                         hover_color=theme.ACCENT_SOFT, border_width=1, border_color=theme.BORDER, **kwargs)
