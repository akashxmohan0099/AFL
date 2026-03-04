"""Generate ABOUT.pdf from ABOUT.md using fpdf2 with Unicode TTF fonts."""
import re
from fpdf import FPDF

MD_PATH = "app/ABOUT.md"
PDF_PATH = "app/ABOUT.pdf"

# Colours
DARK_BLUE = (0, 51, 102)
MID_BLUE = (0, 77, 153)
DARK_GREY = (26, 26, 26)
LIGHT_GREY = (240, 240, 240)
TABLE_HEADER_BG = (0, 51, 102)
TABLE_HEADER_FG = (255, 255, 255)
TABLE_ALT_ROW = (245, 245, 245)
WHITE = (255, 255, 255)
RULE_GREY = (200, 200, 200)

# macOS system TTF paths
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
ARIAL_BI = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
COURIER = "/System/Library/Fonts/Supplemental/Courier New.ttf"
COURIER_BOLD = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

# Font family names we'll use
BODY = "Arial"
MONO = "CourierNew"


class AFLPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font(BODY, "I", 8)
            self.set_text_color(*MID_BLUE)
            self.cell(0, 8, "AFL Player Prediction Pipeline", align="L")
            self.ln(4)
            self.set_draw_color(*RULE_GREY)
            self.line(10, self.get_y(), self.w - 10, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(BODY, "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_md(path):
    """Parse the markdown into a list of blocks."""
    with open(path) as f:
        lines = f.readlines()

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Headings
        m = re.match(r'^(#{1,3})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            blocks.append({"type": f"h{level}", "text": m.group(2)})
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].rstrip("\n").startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1  # skip closing ```
            blocks.append({"type": "code", "text": "\n".join(code_lines)})
            continue

        # Table
        if "|" in stripped and i + 1 < len(lines) and re.match(r'^[\|\s\-:]+$', lines[i + 1].strip()):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i].rstrip("\n"))
                i += 1
            header = [c.strip() for c in table_lines[0].strip("|").split("|")]
            rows = []
            for tl in table_lines[2:]:
                rows.append([c.strip() for c in tl.strip("|").split("|")])
            blocks.append({"type": "table", "header": header, "rows": rows})
            continue

        # Blank line
        if stripped == "":
            i += 1
            continue

        # Regular paragraph
        para_lines = []
        while i < len(lines):
            s = lines[i].rstrip("\n")
            if s == "" or s.startswith("#") or s.startswith("```") or s == "---":
                break
            if "|" in s and i + 1 < len(lines) and re.match(r'^[\|\s\-:]+$', lines[i + 1].strip()):
                break
            para_lines.append(s)
            i += 1
        text = " ".join(para_lines)
        blocks.append({"type": "paragraph", "text": text})

    return blocks


def clean_inline(text):
    """Strip markdown inline formatting for plain text rendering."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


def render_rich_text(pdf, text, default_size=10):
    """Render text with bold and code segments inline."""
    parts = re.split(r'(\*\*.*?\*\*|`.*?`)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font(BODY, "B", default_size)
            pdf.set_text_color(*DARK_BLUE)
            pdf.write(5, part[2:-2])
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font(MONO, "", default_size - 1)
            pdf.set_text_color(80, 80, 80)
            pdf.write(5, part[1:-1])
        else:
            pdf.set_font(BODY, "", default_size)
            pdf.set_text_color(*DARK_GREY)
            pdf.write(5, part)
    pdf.ln(6)


def render_table(pdf, header, rows):
    """Render a table with header and alternating row colours."""
    n_cols = len(header)
    usable = pdf.w - 20

    # Column widths proportional to content length
    col_widths = []
    for ci in range(n_cols):
        max_len = len(header[ci]) if ci < len(header) else 5
        for row in rows:
            if ci < len(row):
                max_len = max(max_len, len(clean_inline(row[ci])))
        col_widths.append(max_len)
    total = sum(col_widths) or 1
    col_widths = [(w / total) * usable for w in col_widths]
    col_widths = [max(w, 15) for w in col_widths]
    scale = usable / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    def draw_header():
        pdf.set_font(BODY, "B", 8)
        pdf.set_fill_color(*TABLE_HEADER_BG)
        pdf.set_text_color(*TABLE_HEADER_FG)
        for ci, h in enumerate(header):
            w = col_widths[ci] if ci < len(col_widths) else 20
            pdf.cell(w, 7, clean_inline(h)[:60], border=1, fill=True)
        pdf.ln()

    draw_header()

    pdf.set_font(BODY, "", 8)
    for ri, row in enumerate(rows):
        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()
            draw_header()
            pdf.set_font(BODY, "", 8)

        if ri % 2 == 1:
            pdf.set_fill_color(*TABLE_ALT_ROW)
        else:
            pdf.set_fill_color(*WHITE)
        pdf.set_text_color(*DARK_GREY)
        for ci in range(n_cols):
            w = col_widths[ci] if ci < len(col_widths) else 20
            val = clean_inline(row[ci]) if ci < len(row) else ""
            max_chars = int(w / 1.6)
            if len(val) > max_chars:
                val = val[:max_chars - 2] + ".."
            pdf.cell(w, 6, val, border=1, fill=True)
        pdf.ln()
    pdf.ln(3)


def build_pdf(blocks, output_path):
    pdf = AFLPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Register Unicode TTF fonts
    pdf.add_font(BODY, "", ARIAL)
    pdf.add_font(BODY, "B", ARIAL_BOLD)
    pdf.add_font(BODY, "I", ARIAL_ITALIC)
    pdf.add_font(BODY, "BI", ARIAL_BI)
    pdf.add_font(MONO, "", COURIER)
    pdf.add_font(MONO, "B", COURIER_BOLD)

    pdf.add_page()

    for block in blocks:
        btype = block["type"]

        if btype == "h1":
            pdf.ln(6)
            pdf.set_font(BODY, "B", 22)
            pdf.set_text_color(*DARK_BLUE)
            pdf.multi_cell(0, 10, clean_inline(block["text"]))
            pdf.set_draw_color(*DARK_BLUE)
            pdf.set_line_width(0.6)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(6)

        elif btype == "h2":
            if pdf.get_y() > pdf.h - 50:
                pdf.add_page()
            pdf.ln(4)
            pdf.set_font(BODY, "B", 16)
            pdf.set_text_color(*MID_BLUE)
            pdf.multi_cell(0, 8, clean_inline(block["text"]))
            pdf.set_draw_color(*RULE_GREY)
            pdf.set_line_width(0.3)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(4)

        elif btype == "h3":
            pdf.ln(3)
            pdf.set_font(BODY, "B", 12)
            pdf.set_text_color(51, 51, 51)
            pdf.multi_cell(0, 7, clean_inline(block["text"]))
            pdf.ln(2)

        elif btype == "hr":
            pdf.ln(4)
            pdf.set_draw_color(*RULE_GREY)
            pdf.set_line_width(0.2)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(4)

        elif btype == "code":
            pdf.ln(2)
            pdf.set_fill_color(*LIGHT_GREY)
            pdf.set_font(MONO, "", 8)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(12)
            for code_line in block["text"].split("\n"):
                if pdf.get_y() > pdf.h - 25:
                    pdf.add_page()
                pdf.cell(pdf.w - 24, 4.5, code_line, fill=True,
                         new_x="LMARGIN", new_y="NEXT")
                pdf.set_x(12)
            pdf.ln(3)

        elif btype == "table":
            pdf.ln(2)
            render_table(pdf, block["header"], block["rows"])

        elif btype == "paragraph":
            if pdf.get_y() > pdf.h - 25:
                pdf.add_page()
            render_rich_text(pdf, block["text"])

    pdf.output(output_path)
    print(f"PDF written to {output_path}")


if __name__ == "__main__":
    blocks = parse_md(MD_PATH)
    build_pdf(blocks, PDF_PATH)
