"""
Pure Python 2-Page PDF Generator for Kuku Solution Document
============================================================
Builds a high-quality, professional 2-page PDF conforming to PDF 1.4 specs
using standard Helvetica/Helvetica-Bold fonts without any third-party libraries.
"""

import os
import re

class PDFCanvas:
    def __init__(self, width=612.0, height=792.0):  # Standard Letter size
        self.width = width
        self.height = height
        self.pages = []  # list of list of content stream strings
        self.current_page = []

    def new_page(self):
        if self.current_page:
            self.pages.append("\n".join(self.current_page))
            self.current_page = []

    def finish(self):
        if self.current_page:
            self.pages.append("\n".join(self.current_page))
            self.current_page = []

    def set_color_rgb(self, r, g, b, stroke=False):
        if stroke:
            self.current_page.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        else:
            self.current_page.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def draw_rect(self, x, y, w, h, fill=True, stroke=False, line_width=1.0):
        if line_width != 1.0:
            self.current_page.append(f"{line_width:.2f} w")
        op = "B" if (fill and stroke) else ("f" if fill else "S")
        self.current_page.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def draw_line(self, x1, y1, x2, y2, line_width=1.0):
        self.current_page.append(f"{line_width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def draw_text(self, text, x, y, font="/F1", size=10.0, r=0.1, g=0.1, b=0.1):
        # Escape special PDF characters
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.current_page.append(
            f"BT {font} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({safe_text}) Tj ET"
        )

    def to_bytes(self):
        self.finish()
        objects = []
        
        # 1: Catalog
        # 2: Pages
        # Page objects: 3, 4 ...
        # Content stream objects: 5, 6 ...
        
        num_pages = len(self.pages)
        page_obj_ids = [3 + i for i in range(num_pages)]
        content_obj_ids = [3 + num_pages + i for i in range(num_pages)]
        font_obj_id = 3 + 2 * num_pages
        font_bold_obj_id = font_obj_id + 1
        font_mono_obj_id = font_bold_obj_id + 1

        # Catalog
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        
        # Pages object
        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects.append(f"<< /Type /Pages /Kids [ {kids_str} ] /Count {num_pages} >>".encode('ascii'))

        # Page objects
        for i in range(num_pages):
            p_obj = (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [ 0 0 {self.width:.2f} {self.height:.2f} ] "
                f"/Contents {content_obj_ids[i]} 0 R "
                f"/Resources << /Font << /F1 {font_obj_id} 0 R /F2 {font_bold_obj_id} 0 R /F3 {font_mono_obj_id} 0 R >> >> >>"
            )
            objects.append(p_obj.encode('ascii'))

        # Content streams
        for i in range(num_pages):
            content_bytes = self.pages[i].encode('utf-8', errors='ignore')
            stream_obj = (
                f"<< /Length {len(content_bytes)} >>\nstream\n".encode('ascii')
                + content_bytes
                + b"\nendstream"
            )
            objects.append(stream_obj)

        # Fonts (Helvetica, Helvetica-Bold, Courier)
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

        # Build PDF with xref
        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        
        xref_offsets = []
        for idx, obj in enumerate(objects, 1):
            xref_offsets.append(len(pdf))
            pdf.extend(f"{idx} 0 obj\n".encode('ascii'))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")

        startxref = len(pdf)
        pdf.extend(b"xref\n")
        pdf.extend(f"0 {len(objects) + 1}\n".encode('ascii'))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in xref_offsets:
            pdf.extend(f"{offset:010d} 00000 n \n".encode('ascii'))

        pdf.extend(b"trailer\n")
        pdf.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode('ascii'))
        pdf.extend(b"startxref\n")
        pdf.extend(f"{startxref}\n%%EOF\n".encode('ascii'))

        return bytes(pdf)


def wrap_text(text, max_chars=88):
    words = text.split()
    lines = []
    current = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 <= max_chars:
            current.append(w)
            cur_len += len(w) + 1
        else:
            lines.append(" ".join(current))
            current = [w]
            cur_len = len(w)
    if current:
        lines.append(" ".join(current))
    return lines


def generate_solution_pdf(output_path="Solution_Doc_Sanket_Jambhule.pdf"):
    canvas = PDFCanvas(width=612.0, height=792.0)
    
    # =========================================================================
    # PAGE 1
    # =========================================================================
    
    # Top banner background
    canvas.set_color_rgb(0.96, 0.97, 0.99)
    canvas.draw_rect(36, 715, 540, 52, fill=True, stroke=False)
    canvas.set_color_rgb(0.88, 0.12, 0.28, stroke=True)
    canvas.draw_line(36, 715, 576, 715, line_width=2.5)

    # Title & Subtitle
    canvas.draw_text("KUKU AI GENERALIST ASSIGNMENT - SOLUTION DOC", 44, 750, font="/F2", size=13.0, r=0.08, g=0.12, b=0.2)
    canvas.draw_text("Problem 1: Find the Promos | Two-Stage Hybrid Extraction Pipeline", 44, 735, font="/F1", size=8.5, r=0.3, g=0.35, b=0.45)
    
    # Metadata Right
    canvas.draw_text("Candidate: Sanket Jambhule", 410, 750, font="/F2", size=8.5, r=0.1, g=0.15, b=0.25)
    canvas.draw_text("Repo: github.com/sanketjambhule/kuku-promo-tool", 355, 736, font="/F1", size=7.8, r=0.2, g=0.3, b=0.6)
    canvas.draw_text("Prototype: Streamlit Community Cloud", 410, 723, font="/F1", size=7.8, r=0.3, g=0.4, b=0.5)

    y = 698

    # Section 1: Problem Framing
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 230, y+2, line_width=1.0)
    canvas.draw_text("1. PROBLEM FRAMING & STRATEGIC CONTEXT", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    p1 = (
        "Microdrama platforms (Kuku, ReelShort, DramaBox) scale through continuous, high-volume ad creative testing. "
        "User acquisition hinges entirely on 20 to 60-second vertical promotional video ads posted across TikTok, Reels, and Shorts. "
        "Today, human editors watch full episodes to manually cut promos - a bottleneck that breaks when releasing multiple shows weekly."
    )
    for line in wrap_text(p1, 95):
        canvas.draw_text(line, 36, y, font="/F1", size=8.5, r=0.15, g=0.15, b=0.15)
        y -= 11

    y -= 2
    canvas.draw_text("We framed promo extraction as an Attention Economics Optimization problem across three non-negotiables:", 36, y, font="/F2", size=8.5, r=0.1, g=0.1, b=0.1)
    y -= 11

    bullets_p1 = [
        ("The 0-3s Hook:", "Must arrest passive infinite scrolling with immediate conflict, a physical slap, or an outrageous accusation."),
        ("The Final 3s Cliffhanger:", "Must cut precisely at peak unresolved emotional tension, forcing viewers to click to Episode 1."),
        ("Standalone Comprehension:", "Cold scrolling audiences with zero prior lore must instantly understand who the victim and villain are.")
    ]
    for b_title, b_desc in bullets_p1:
        canvas.draw_text(f"- {b_title}", 44, y, font="/F2", size=8.2, r=0.1, g=0.1, b=0.1)
        canvas.draw_text(b_desc, 155, y, font="/F1", size=8.2, r=0.2, g=0.2, b=0.2)
        y -= 10.5

    y -= 6

    # Section 2: Architecture
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 255, y+2, line_width=1.0)
    canvas.draw_text("2. TWO-STAGE HYBRID PIPELINE ARCHITECTURE", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    p2 = (
        "Brute-force LLM evaluation of all sliding windows in a 2-minute episode (~250 windows) costs ~$1.50/ep and takes 60s+. "
        "Across a 100-episode season, this is financially unsustainable. We designed a Two-Stage Hybrid Engine:"
    )
    for line in wrap_text(p2, 95):
        canvas.draw_text(line, 36, y, font="/F1", size=8.5, r=0.15, g=0.15, b=0.15)
        y -= 11

    # Diagram Box
    y -= 2
    canvas.set_color_rgb(0.08, 0.11, 0.18)
    canvas.draw_rect(36, y-18, 540, 24, fill=True, stroke=False)
    canvas.draw_text("[Screenplay .txt] -> [Parser] -> [Sliding Windows 20-60s] -> [Stage 1 Heuristic + NMS] -> Top 10 -> [Stage 2 LLM] -> [Ranker]", 42, y-6, font="/F3", size=7.2, r=0.9, g=0.95, b=1.0)
    y -= 28

    canvas.draw_text("Stage 1: Deterministic Heuristic Filter (<50ms execution in Python)", 36, y, font="/F2", size=8.8, r=0.1, g=0.1, b=0.1)
    y -= 11

    s1_bullets = [
        "- Beat Extraction & Timestamp Normalization: Parses scenes, character dialogue, actions, and camera cues.",
        "- Missing Timestamp Interpolation: Automatically estimates speech rate (2.5 words/sec, min 2s) if timestamps are missing.",
        "- Virality Lexicon Scoring: Scans S-Tier Drama (slap, marry, betray, kill, bastard), A-Tier (liar, kneel, revenge), B-Tier (desperate).",
        "- Hook & Cliffhanger Structural Check: Rewards dialogue/action opening in first 4s and unresolved endings (?, !, ...).",
        "- Temporal Non-Maximum Suppression (NMS): Suppresses windows with IoU overlap > 65% to guarantee diverse scene coverage."
    ]
    for b in s1_bullets:
        canvas.draw_text(b, 44, y, font="/F1", size=8.0, r=0.2, g=0.2, b=0.2)
        y -= 10.0

    y -= 4
    canvas.draw_text("Stage 2: Semantic LLM Evaluator (Single Batched Call via Pydantic Schema)", 36, y, font="/F2", size=8.8, r=0.1, g=0.1, b=0.1)
    y -= 11
    p3 = (
        "The filtered Top 10 candidate windows are passed in a single structured prompt to Google Gemini 1.5 Flash or Claude 3 Haiku. "
        "The model analyzes subtle character status dynamics, emotional escalation, and outputs strict Pydantic JSON."
    )
    for line in wrap_text(p3, 95):
        canvas.draw_text(line, 44, y, font="/F1", size=8.2, r=0.2, g=0.2, b=0.2)
        y -= 10.5

    y -= 6

    # Section 3: Ranking Formula
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 240, y+2, line_width=1.0)
    canvas.draw_text("3. MULTI-CRITERIA COMPOSITE RANKING FORMULA", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    # Formula Box
    canvas.set_color_rgb(0.95, 0.96, 0.98)
    canvas.draw_rect(36, y-15, 540, 20, fill=True, stroke=False)
    canvas.draw_text("Viral Score = 2.0 x [ (0.40 x Hook Rating) + (0.40 x Cliffhanger Rating) + (0.20 x Standalone Comprehension Score) ]", 52, y-3, font="/F2", size=8.2, r=0.15, g=0.2, b=0.3)
    y -= 23

    p_f = "Ratings range 1-5. Standalone Comprehension maps: High=5.0, Medium=3.0, Low=1.0. Multiplied by 2.0 to scale onto 1.0 - 10.0."
    canvas.draw_text(p_f, 44, y, font="/F1", size=7.8, r=0.3, g=0.3, b=0.3)
    
    # Bottom page 1 marker
    canvas.draw_text("Page 1 of 2", 285, 20, font="/F1", size=8.0, r=0.5, g=0.5, b=0.5)

    # =========================================================================
    # PAGE 2
    # =========================================================================
    canvas.new_page()

    # Top Header P2
    canvas.draw_text("KUKU AI GENERALIST ASSIGNMENT - SOLUTION DOC (PAGE 2)", 36, 755, font="/F2", size=9.0, r=0.3, g=0.35, b=0.45)
    canvas.set_color_rgb(0.88, 0.12, 0.28, stroke=True)
    canvas.draw_line(36, 748, 576, 748, line_width=1.5)

    y = 732

    # Section 4: Key Assumptions
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 175, y+2, line_width=1.0)
    canvas.draw_text("4. KEY ASSUMPTIONS MADE", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    assumptions = [
        ("Screenplay-Video Fidelity:", "Assumes final video edit reflects screenplay dialogue sequence and timestamps."),
        ("Pacing Baseline:", "For un-timed screenplays, dialogue rate is calibrated to 2.5 words/second with a 2.0s floor for non-verbal actions."),
        ("Duration Boundary (20-60s):", "Clips under 20s lack dramatic setup; clips over 60s suffer severe retention drop-off on TikTok/Shorts.")
    ]
    for a_t, a_d in assumptions:
        canvas.draw_text(f"- {a_t}", 44, y, font="/F2", size=8.2, r=0.1, g=0.1, b=0.1)
        canvas.draw_text(a_d, 175, y, font="/F1", size=8.2, r=0.2, g=0.2, b=0.2)
        y -= 10.5

    y -= 6

    # Section 5: Posting Strategy
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 280, y+2, line_width=1.0)
    canvas.draw_text("5. POSTING STRATEGY: WHICH PROMOS TO POST FIRST & WHY?", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    p5 = "Running the pipeline across all 50 episodes yielded 214 ranked promos. We established a 3-tier marketing deployment strategy:"
    canvas.draw_text(p5, 36, y, font="/F1", size=8.5, r=0.15, g=0.15, b=0.15)
    y -= 13

    # Post 1 Box
    canvas.set_color_rgb(0.98, 0.98, 1.0)
    canvas.draw_rect(36, y-34, 540, 42, fill=True, stroke=False)
    canvas.set_color_rgb(0.2, 0.4, 0.8, stroke=True)
    canvas.draw_line(36, y-34, 36, y+8, line_width=2.5)

    canvas.draw_text("POST #1: Primary Paid User Acquisition Ad (Meta & TikTok Ads) -> Episode 1 [00:21.80 -> 01:21.20] (Score: 9.20/10)", 44, y, font="/F2", size=8.0, r=0.1, g=0.2, b=0.5)
    canvas.draw_text("Hook: Matriarch stops bride: 'You should turn around, Juliana' | Cliffhanger: Juliana begs father: 'Father... tell them this is a mistake.'", 44, y-10, font="/F1", size=7.5, r=0.2, g=0.2, b=0.2)
    canvas.draw_text("Why First: High Standalone Comprehension. Cold paid traffic will not tolerate complex backstories; a bride humiliated at the altar for her cruel", 44, y-20, font="/F1", size=7.5, r=0.3, g=0.3, b=0.3)
    canvas.draw_text("sister creates instant emotional outrage that directly drives app installs.", 44, y-29, font="/F1", size=7.5, r=0.3, g=0.3, b=0.3)
    y -= 44

    # Post 2 Box
    canvas.set_color_rgb(0.99, 0.98, 0.98)
    canvas.draw_rect(36, y-34, 540, 42, fill=True, stroke=False)
    canvas.set_color_rgb(0.85, 0.2, 0.2, stroke=True)
    canvas.draw_line(36, y-34, 36, y+8, line_width=2.5)

    canvas.draw_text("POST #2: Organic Viral Hook (TikTok / Reels Algorithmic Loop) -> Episode 42 [00:18.00 -> 00:39.90] (Score: 8.80/10)", 44, y, font="/F2", size=8.0, r=0.5, g=0.1, b=0.1)
    canvas.draw_text("Hook: Rupert shrieks: 'Guards! Arrest this imposter!' | Cliffhanger: Royal Guard: 'We will not... because we know who he is.'", 44, y-10, font="/F1", size=7.5, r=0.2, g=0.2, b=0.2)
    canvas.draw_text("Why Second: 5/5 Hook Shock Value. Sudden defiance of authority triggers massive organic replay loops and comment debates, driving free viral", 44, y-20, font="/F1", size=7.5, r=0.3, g=0.3, b=0.3)
    canvas.draw_text("reach even among casual non-subscribers.", 44, y-29, font="/F1", size=7.5, r=0.3, g=0.3, b=0.3)
    y -= 44

    # Post 3 Box
    canvas.set_color_rgb(0.98, 0.99, 0.98)
    canvas.draw_rect(36, y-30, 540, 38, fill=True, stroke=False)
    canvas.set_color_rgb(0.15, 0.6, 0.3, stroke=True)
    canvas.draw_line(36, y-30, 36, y+8, line_width=2.5)

    canvas.draw_text("POST #3: Mid-Funnel Retargeting (Cliffhanger Conversion) -> Episode 34 [00:13.10 -> 00:44.80] (Score: 8.60/10)", 44, y, font="/F2", size=8.0, r=0.1, g=0.4, b=0.2)
    canvas.draw_text("Hook: Witchcraft accusation in dock | Cliffhanger: Maid Prudence steps onto witness stand weeping to give coerced false testimony.", 44, y-10, font="/F1", size=7.5, r=0.2, g=0.2, b=0.2)
    canvas.draw_text("Why Third: Maximum cliffhanger urgency (5/5). Converts warm users who watched an earlier ad teaser without completing the install.", 44, y-20, font="/F1", size=7.5, r=0.3, g=0.3, b=0.3)
    y -= 40

    # Section 6: Quality Evaluation & Failure Modes
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 285, y+2, line_width=1.0)
    canvas.draw_text("6. QUALITY CHECKING: WHERE DOES THE TOOL PICK BADLY?", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    failures = [
        ("Subtle Irony & Deadpan Sarcasm:", "Keywords flag overt conflict. A whisper-delivered line like 'How delightful to see you, sister' scores lower in Stage 1 keyword density despite being a brilliant psychological hook."),
        ("Visual-Only Action Beats:", "A silent slap or poison-drop described in CAMERA cues can be penalized by the dialogue-density rule unless explicitly written as an ACTION beat."),
        ("Mid-Sentence Scene Boundaries:", "Windows spanning scene cuts occasionally slice midway through a multi-beat sentence, requiring video audio-cut crossfades.")
    ]
    for f_t, f_d in failures:
        canvas.draw_text(f"- {f_t}", 44, y, font="/F2", size=8.0, r=0.1, g=0.1, b=0.1)
        for sub_line in wrap_text(f_d, 72):
            canvas.draw_text(sub_line, 185, y, font="/F1", size=8.0, r=0.2, g=0.2, b=0.2)
            y -= 9.5
        y -= 2

    y -= 4

    # Section 7: Scaling to Every Show
    canvas.set_color_rgb(0.9, 0.15, 0.25, stroke=True)
    canvas.draw_line(36, y+2, 215, y+2, line_width=1.0)
    canvas.draw_text("7. PRODUCTION SCALING & ECONOMICS", 36, y+5, font="/F2", size=9.5, r=0.08, g=0.12, b=0.2)
    y -= 12

    scale_pts = [
        "- Cost Efficiency: Brute-force LLM calls cost ~$1.50/ep ($150/season). Two-Stage Hybrid costs <$0.002/ep ($0.18/season) - a 99.8% cost reduction.",
        "- High Throughput: Batch processes an entire 50-episode catalog in under 2 minutes (214 promos generated).",
        "- Multimodal Expansion Roadmap: Feed rendered MP4s into Gemini 2.0 Flash Video API to score facial expressions and music swell alongside text."
    ]
    for pt in scale_pts:
        canvas.draw_text(pt, 44, y, font="/F1", size=8.0, r=0.15, g=0.15, b=0.15)
        y -= 10.0

    # Bottom page 2 marker
    canvas.draw_text("Page 2 of 2", 285, 20, font="/F1", size=8.0, r=0.5, g=0.5, b=0.5)

    # Save to disk
    pdf_bytes = canvas.to_bytes()
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"SUCCESS! Wrote {len(pdf_bytes)} bytes to {output_path}")

if __name__ == "__main__":
    generate_solution_pdf("Solution_Doc_Sanket_Jambhule.pdf")
