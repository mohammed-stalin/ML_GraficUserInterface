"""PDF report of the dataset, the cleaning steps and the latest model results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ACCENT = colors.HexColor("#2563EB")
MAX_TABLE_ROWS = 60


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1e6 else f"{value:,.0f}"
    return str(value)


def _table(df: pd.DataFrame, width: float) -> Table:
    styles = getSampleStyleSheet()
    cell = styles["BodyText"].clone("cell", fontSize=7.5, leading=9)
    head = cell.clone("head", textColor=colors.white, fontName="Helvetica-Bold")
    shown = df.head(MAX_TABLE_ROWS)
    data = [[Paragraph(str(c), head) for c in shown.columns]]
    data += [[Paragraph(_fmt(v), cell) for v in row] for row in shown.itertuples(index=False)]
    table = Table(data, colWidths=[width / max(len(shown.columns), 1)] * len(shown.columns), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def build_report(path: str | Path, dataset, classification=None, regression=None, clustering=None) -> Path:
    path = Path(path)
    page = landscape(A4)
    doc = SimpleDocTemplate(str(path), pagesize=page, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="ML Studio report")
    width = page[0] - 3 * cm
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    story = [
        Paragraph("ML Studio report", h1),
        Paragraph(f"Dataset: <b>{dataset.name}</b> &nbsp;·&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}", body),
        Spacer(1, 12),
    ]
    summary = dataset.summary()
    story += [Paragraph("Overview", h2), _table(pd.DataFrame(
        [{k.replace("_", " ").capitalize(): v for k, v in summary.items()}]), width), Spacer(1, 10)]
    story += [Paragraph("Columns", h2), _table(dataset.profile(), width), Spacer(1, 10)]
    describe = dataset.describe()
    if not describe.empty:
        story += [Paragraph("Numeric statistics", h2), _table(describe, width), Spacer(1, 10)]

    story.append(Paragraph("Preparation steps", h2))
    if dataset.log:
        story += [Paragraph(f"{i}. {step}", body) for i, step in enumerate(dataset.log, 1)]
    else:
        story.append(Paragraph("No cleaning operations were applied.", body))

    if classification is not None:
        r = classification
        story += [Spacer(1, 10), Paragraph(f"Classification — {r.model_name} (target: {r.target})", h2),
                  Paragraph(f"Accuracy {r.accuracy:.3f} · balanced accuracy {r.balanced_accuracy:.3f} · "
                            f"macro F1 {r.f1_macro:.3f} · CV accuracy {r.cv_accuracy:.3f} ± {r.cv_accuracy_std:.3f} "
                            f"({r.train_rows} train / {r.test_rows} test rows)", body),
                  Spacer(1, 6), _table(r.report.round(3), width)]
    if regression is not None:
        story += [Spacer(1, 10), Paragraph(f"Regression — target: {regression.target}", h2),
                  Paragraph(f"Best model: <b>{regression.best_model}</b>"
                            f"{' (log-scale target)' if regression.log_target else ''}", body),
                  Spacer(1, 6), _table(regression.results, width)]
    if clustering is not None:
        sil = f"{clustering.silhouette:.3f}" if clustering.silhouette is not None else "n/a"
        story += [Spacer(1, 10), Paragraph(f"K-Means clustering — k = {clustering.k}", h2),
                  Paragraph(f"Silhouette score {sil}", body), Spacer(1, 6), _table(clustering.profile, width)]
    doc.build(story)
    return path
