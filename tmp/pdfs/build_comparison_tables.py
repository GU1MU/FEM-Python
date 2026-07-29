# -*- coding: utf-8 -*-

import csv
import math
import os
from itertools import zip_longest

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRAM_DIR = os.path.join(ROOT, "results", "complicate_cae_model")
ABAQUS_DIR = os.path.join(ROOT, "results", "complicate_cae_model_odb")
OUTPUT_DIR = os.path.join(ROOT, "output", "pdf")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "complicate_cae_model_numerical_comparison.pdf"
)
FONT_NAME = "MicrosoftYaHei"


class Metric(object):
    def __init__(self):
        self.abaqus_min = float("inf")
        self.abaqus_max = float("-inf")
        self.program_min = float("inf")
        self.program_max = float("-inf")
        self.difference_square_sum = 0.0
        self.abaqus_square_sum = 0.0

    def add(self, program_value, abaqus_value):
        self.abaqus_min = min(self.abaqus_min, abaqus_value)
        self.abaqus_max = max(self.abaqus_max, abaqus_value)
        self.program_min = min(self.program_min, program_value)
        self.program_max = max(self.program_max, program_value)
        self.difference_square_sum += (program_value - abaqus_value) ** 2
        self.abaqus_square_sum += abaqus_value ** 2

    @property
    def error(self):
        return math.sqrt(
            self.difference_square_sum / self.abaqus_square_sum
        )


def compare_displacement():
    names = ("UMAG", "U1", "U2", "U3")
    metrics = dict((name, Metric()) for name in names)
    program_path = os.path.join(
        PROGRAM_DIR, "ComplicateCAEModel_nodal_displacement.csv"
    )
    abaqus_path = os.path.join(
        ABAQUS_DIR, "ComplicateCAEModel_nodal_displacement.csv"
    )
    count = 0

    with open(program_path, encoding="utf-8", newline="") as program_file:
        with open(abaqus_path, encoding="utf-8", newline="") as abaqus_file:
            program_rows = csv.DictReader(program_file)
            abaqus_rows = csv.DictReader(abaqus_file)
            for program_row, abaqus_row in zip_longest(
                program_rows, abaqus_rows
            ):
                if (
                    program_row is None
                    or abaqus_row is None
                    or program_row["node_id"] != abaqus_row["node_id"]
                ):
                    raise ValueError("位移数据不能逐节点对应")
                program = [
                    float(program_row[name]) for name in ("ux", "uy", "uz")
                ]
                abaqus = [
                    float(abaqus_row[name]) for name in ("ux", "uy", "uz")
                ]
                metrics["UMAG"].add(
                    math.sqrt(sum(value * value for value in program)),
                    math.sqrt(sum(value * value for value in abaqus)),
                )
                for name, program_value, abaqus_value in zip(
                    ("U1", "U2", "U3"), program, abaqus
                ):
                    metrics[name].add(program_value, abaqus_value)
                count += 1
    return names, metrics, count


def compare_stress():
    names = ("Mises", "S11", "S22", "S33", "S12", "S13", "S23")
    metrics = dict((name, Metric()) for name in names)
    program_path = os.path.join(
        PROGRAM_DIR, "ComplicateCAEModel_nodal_stress.csv"
    )
    abaqus_path = os.path.join(
        ABAQUS_DIR, "ComplicateCAEModel_nodal_stress.csv"
    )
    count = 0

    with open(program_path, encoding="utf-8", newline="") as program_file:
        with open(abaqus_path, encoding="utf-8", newline="") as abaqus_file:
            program_rows = csv.DictReader(program_file)
            abaqus_rows = csv.DictReader(abaqus_file)
            for program_row, abaqus_row in zip_longest(
                program_rows, abaqus_rows
            ):
                if (
                    program_row is None
                    or abaqus_row is None
                    or program_row["node_id"] != abaqus_row["node_id"]
                ):
                    raise ValueError("应力数据不能按原始行序对应")
                for name in names:
                    metrics[name].add(
                        float(program_row[name]),
                        float(abaqus_row[name]),
                    )
                count += 1
    return names, metrics, count


def scientific(value):
    return "%.6e" % value


def make_table(names, metrics, body_style):
    rows = [
        [
            "",
            "",
            Paragraph("Abaqus", body_style),
            Paragraph("程序", body_style),
            Paragraph("全场误差", body_style),
        ]
    ]
    spans = [("SPAN", (0, 0), (1, 0))]
    max_rows = []
    min_rows = []

    for name in names:
        metric = metrics[name]
        max_row = len(rows)
        min_row = max_row + 1
        rows.extend(
            [
                [
                    Paragraph(name, body_style),
                    Paragraph("Max", body_style),
                    scientific(metric.abaqus_max),
                    scientific(metric.program_max),
                    scientific(metric.error),
                ],
                [
                    "",
                    Paragraph("Min", body_style),
                    scientific(metric.abaqus_min),
                    scientific(metric.program_min),
                    "",
                ],
            ]
        )
        spans.extend(
            [
                ("SPAN", (0, max_row), (0, min_row)),
                ("SPAN", (4, max_row), (4, min_row)),
            ]
        )
        max_rows.append(max_row)
        min_rows.append(min_row)

    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
        ("LINEAFTER", (1, 0), (1, -1), 0.45, colors.black),
        ("LINEAFTER", (2, 0), (2, -1), 0.45, colors.black),
        ("LINEAFTER", (3, 0), (3, -1), 0.45, colors.black),
        ("LINEAFTER", (0, 1), (0, -1), 0.45, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]
    for max_row in max_rows:
        style.append(
            ("LINEBELOW", (1, max_row), (3, max_row), 0.35, colors.black)
        )
    for min_row in min_rows:
        style.append(
            ("LINEBELOW", (0, min_row), (-1, min_row), 0.45, colors.black)
        )
    style.extend(spans)

    table = Table(
        rows,
        colWidths=(28 * mm, 22 * mm, 39 * mm, 39 * mm, 35 * mm),
        rowHeights=[6.5 * mm] + [5.8 * mm] * (2 * len(names)),
    )
    table.setStyle(TableStyle(style))
    return table


def page_number(canvas, document):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2.0, 11 * mm, str(document.page))
    canvas.restoreState()


def build_pdf(displacement, stress):
    displacement_names, displacement_metrics, displacement_count = displacement
    stress_names, stress_metrics, stress_count = stress

    pdfmetrics.registerFont(
        TTFont(FONT_NAME, r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0)
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    styles = getSampleStyleSheet()
    section_style = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=13,
        leading=17,
        spaceAfter=6,
    )
    formula_style = ParagraphStyle(
        "formula",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.5,
        textColor=colors.red,
        alignment=TA_CENTER,
        leading=15,
        spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "caption",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=10,
        alignment=TA_CENTER,
        leading=14,
        spaceAfter=1.5 * mm,
    )
    body_style = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
    )
    note_style = ParagraphStyle(
        "note",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#333333"),
        spaceBefore=4 * mm,
    )

    story = [
        Paragraph("3. 数值比较", section_style),
        Paragraph(
            "error = ||X<sub>程序</sub> - X<sub>Abaqus</sub>||"
            "<sub>2</sub> / ||X<sub>Abaqus</sub>||<sub>2</sub>",
            formula_style,
        ),
        Paragraph("[节点位移对比]", caption_style),
        make_table(displacement_names, displacement_metrics, body_style),
        Spacer(1, 5 * mm),
        Paragraph("[原始单元节点应力对比]", caption_style),
        make_table(stress_names, stress_metrics, body_style),
        Paragraph(
            "数据说明：节点位移共 %s 条，原始单元节点应力共 %s 条。"
            "程序结果与 Abaqus 结果分别取自 complicate_cae_model 和 "
            "complicate_cae_model_odb，两组数据按原始行序和 node_id 对齐。"
            % (format(displacement_count, ","), format(stress_count, ",")),
            note_style,
        ),
    ]

    document = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=23 * mm,
        rightMargin=23 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="ComplicateCAEModel 数值比较",
        author="FEM-Python",
    )
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


build_pdf(compare_displacement(), compare_stress())
print(OUTPUT_PATH)
