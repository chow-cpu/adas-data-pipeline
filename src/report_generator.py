import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io
import os

OUTPUT_PATH = "output/adas_report.pdf"

def fig_to_image(fig):
    """Convert a matplotlib figure to a reportlab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def generate_report(vehicles, vehicle_ids, health_scores, zscore_outliers_dict, ml_outliers_dict, channel):
    """Generate a professional PDF report for the ADAS simulation run."""
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#1F3864"),
        spaceAfter=6, alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#1F3864"),
        spaceBefore=12, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        "Normal", parent=styles["Normal"],
        fontSize=10, spaceAfter=4
    )

    story = []

    # Title
    story.append(Paragraph("ADAS Sensor Data Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                           ParagraphStyle("sub", parent=styles["Normal"], 
                                        alignment=TA_CENTER, textColor=colors.grey)))
    story.append(Spacer(1, 0.3*inch))

    # Fleet summary table
    story.append(Paragraph("Fleet Health Summary", heading_style))

    table_data = [["Vehicle", "ID", "Avg Speed", "Max Speed", "Z-Score Anomalies", "ML Anomalies", "Health Score", "Status"]]

    for name, df in vehicles.items():
        vid = vehicle_ids[name]
        score, status, _ = health_scores[name]
        z_count = len(zscore_outliers_dict[name])
        ml_count = len(ml_outliers_dict[name])

        table_data.append([
            name.split("—")[1].strip(),
            vid,
            f"{df['speed_mps'].mean():.2f} m/s",
            f"{df['speed_mps'].max():.2f} m/s",
            str(z_count),
            str(ml_count),
            f"{score}/100",
            status
        ])

    t = Table(table_data, colWidths=[1.0*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.7*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # Sensor charts
    story.append(Paragraph(f"Sensor Data — {channel}", heading_style))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plot_colors = ["steelblue", "green", "orange"]

    for ax, (name, df), color in zip(axes, vehicles.items(), plot_colors):
        z_out = zscore_outliers_dict[name]
        ax.plot(df["timestamp"], df[channel], color=color, linewidth=0.8)
        if len(z_out) > 0:
            ax.scatter(z_out["timestamp"], z_out[channel], color="red", s=20, zorder=5)
        ax.set_title(name.split("—")[1].strip(), fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel(channel, fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    buf = fig_to_image(fig)
    plt.close()
    story.append(Image(buf, width=6.5*inch, height=2*inch))
    story.append(Spacer(1, 0.2*inch))

    # ML vs Z-score comparison
    story.append(Paragraph("ML vs Z-Score Anomaly Detection", heading_style))

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 5))
    for i, (name, df) in enumerate(vehicles.items()):
        color = plot_colors[i]
        z_out = zscore_outliers_dict[name]
        ml_out = ml_outliers_dict[name]

        axes2[0][i].plot(df["timestamp"], df[channel], color=color, linewidth=0.8)
        if len(z_out) > 0:
            axes2[0][i].scatter(z_out["timestamp"], z_out[channel], color="red", s=15, zorder=5)
        axes2[0][i].set_title(f"{name.split('—')[1].strip()} — Z-Score", fontsize=7)
        axes2[0][i].tick_params(labelsize=6)

        axes2[1][i].plot(df["timestamp"], df[channel], color=color, linewidth=0.8)
        if len(ml_out) > 0:
            axes2[1][i].scatter(ml_out["timestamp"], ml_out[channel], color="purple", s=15, zorder=5)
        axes2[1][i].set_title(f"{name.split('—')[1].strip()} — ML", fontsize=7)
        axes2[1][i].tick_params(labelsize=6)

    plt.tight_layout()
    buf2 = fig_to_image(fig2)
    plt.close()
    story.append(Image(buf2, width=6.5*inch, height=3.5*inch))
    story.append(Spacer(1, 0.2*inch))

    # Anomaly details
    story.append(Paragraph("Anomaly Details", heading_style))
    for name, df in vehicles.items():
        z_out = zscore_outliers_dict[name]
        if len(z_out) > 0:
            story.append(Paragraph(f"{name} — Z-Score Anomalies ({len(z_out)} found):", normal_style))
            rows = [["Timestamp", "Speed (m/s)", "Latitude", "Longitude"]]
            for _, row in z_out.iterrows():
                rows.append([
                    f"{row['timestamp']:.1f}s",
                    f"{row['speed_mps']:.4f}",
                    f"{row['latitude']:.6f}",
                    f"{row['longitude']:.6f}"
                ])
            t2 = Table(rows, colWidths=[1.2*inch, 1.2*inch, 1.5*inch, 1.5*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EBF3FB")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.1*inch))

    # Footer
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "ADAS Data Pipeline — Automated Simulation Report | Confidential",
        ParagraphStyle("footer", parent=styles["Normal"],
                      fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"Report saved to {OUTPUT_PATH}")
    return OUTPUT_PATH