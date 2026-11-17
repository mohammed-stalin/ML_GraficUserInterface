"""Welcome page with quick start, sample data and credits."""
from __future__ import annotations

import webbrowser

import customtkinter as ctk
from PIL import Image

from .. import theme
from .base import Page, secondary_button

STEPS = [
    ("1", "Load", "Open a CSV or Excel file and review column types, missing values and statistics."),
    ("2", "Prepare", "Fill gaps, encode categories, drop columns, remove duplicates or outliers — with undo."),
    ("3", "Explore", "Histograms, box plots, scatter plots, category counts and correlation heatmaps."),
    ("4", "Model", "Train and compare 8 classifiers, cluster with K-Means, or benchmark regressors."),
    ("5", "Report", "Export a PDF report or save the cleaned dataset."),
]


class HomePage(Page):
    title = "Welcome"
    subtitle = "Explore tabular data and build machine learning models without writing code."
    needs_data = False

    def build(self, body):
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        hero = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=16, border_width=1, border_color=theme.BORDER)
        hero.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(hero, text="Explore data. Compare models.", font=theme.font(28, "bold"),
                     text_color=theme.TEXT, anchor="w").grid(row=0, column=0, sticky="ew", padx=28, pady=(28, 4))
        ctk.CTkLabel(hero, text=("ML Studio guides you from a raw spreadsheet to evaluated models. Every model "
                                 "uses leakage-free preprocessing, cross-validation and clear metrics."),
                     font=theme.font(14), text_color=theme.MUTED, anchor="w", justify="left",
                     wraplength=620).grid(row=1, column=0, sticky="ew", padx=28)
        buttons = ctk.CTkFrame(hero, fg_color="transparent")
        buttons.grid(row=2, column=0, sticky="w", padx=28, pady=20)
        ctk.CTkButton(buttons, text="Open dataset…", height=40, font=theme.font(13, "bold"),
                      command=self.app.open_dataset).pack(side="left")
        sample = self.app.app_dir / "Car Price.csv"
        if sample.exists():
            secondary_button(buttons, "Try sample: car prices", lambda: self.app.open_dataset(str(sample)),
                             height=40).pack(side="left", padx=10)
        manual = self.app.app_dir / "Manuel_of_use.md"
        if manual.exists():
            secondary_button(buttons, "User manual", lambda: webbrowser.open(manual.as_uri()), height=40
                             ).pack(side="left")

        image_path = self.app.app_dir / "home1.png"
        if image_path.exists():
            img = Image.open(image_path)
            ratio = img.height / img.width
            width = 600
            self._image = ctk.CTkImage(light_image=img, dark_image=img, size=(width, int(width * ratio)))
            ctk.CTkLabel(hero, text="", image=self._image).grid(row=3, column=0, sticky="n", padx=28, pady=(0, 20))

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        side.grid_columnconfigure(0, weight=1)
        steps = ctk.CTkFrame(side, fg_color=theme.SURFACE, corner_radius=16, border_width=1, border_color=theme.BORDER)
        steps.grid(row=0, column=0, sticky="ew")
        steps.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(steps, text="How it works", font=theme.font(15, "bold"), text_color=theme.TEXT,
                     anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(18, 8))
        for i, (num, name, text) in enumerate(STEPS, 1):
            ctk.CTkLabel(steps, text=num, width=28, height=28, corner_radius=14, fg_color=theme.ACCENT_SOFT,
                         text_color=theme.ACCENT, font=theme.font(12, "bold")
                         ).grid(row=i, column=0, sticky="n", padx=(20, 12), pady=8)
            block = ctk.CTkFrame(steps, fg_color="transparent")
            block.grid(row=i, column=1, sticky="ew", padx=(0, 20), pady=6)
            ctk.CTkLabel(block, text=name, font=theme.font(13, "bold"), text_color=theme.TEXT, anchor="w").pack(fill="x")
            ctk.CTkLabel(block, text=text, font=theme.font(12), text_color=theme.MUTED, anchor="w", justify="left",
                         wraplength=320).pack(fill="x")
        ctk.CTkFrame(steps, fg_color="transparent", width=1, height=10).grid(row=len(STEPS) + 1, column=0)

        credits = ctk.CTkFrame(side, fg_color=theme.SURFACE, corner_radius=16, border_width=1, border_color=theme.BORDER)
        credits.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        for label, value in (("Developed by", "Mohammed AACHABI & Abdelmajid BENJELLOUN"),
                             ("Supervised by", "Prof. Sanae KHALI ISSA"),
                             ("Programme", "Master AI & Data Science, 2023–2024")):
            ctk.CTkLabel(credits, text=label.upper(), font=theme.font(10, "bold"), text_color=theme.MUTED,
                         anchor="w").pack(fill="x", padx=20, pady=(14, 0))
            ctk.CTkLabel(credits, text=value, font=theme.font(13), text_color=theme.TEXT, anchor="w").pack(fill="x", padx=20)
        ctk.CTkLabel(credits, text="Models can make mistakes. Validate important results.", font=theme.font(11),
                     text_color=theme.MUTED, anchor="w").pack(fill="x", padx=20, pady=(14, 16))
