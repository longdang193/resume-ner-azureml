"""
Script to generate PDF and image files from Resume NER test text lines.
Creates both PDF and PNG files for each text line.
"""

from pathlib import Path
from typing import List

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not found. Install with: pip install reportlab")

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not found. Install with: pip install Pillow")


# Test text lines for Resume NER testing
TEST_TEXTS = [
    "Amazon Web Services, Austin, TX. Lead Machine Learning Engineer with 7 years of experience in NLP, recommender systems, and deep learning. Skills: Python, PyTorch, Spark.",
    "Alice Johnson is a data analyst at Meta Platforms. Email: alice.johnson@meta.com. Phone: +1-408-555-7890. Location: Menlo Park, CA.",
    "Robert Lee holds a PhD in Artificial Intelligence from Stanford University. His expertise includes computer vision, transformers, and TensorFlow.",
    "Netflix Inc., Los Gatos, CA. Senior Software Engineer with 6 years of experience in backend systems and machine learning. Skills: Java, Scala, Kubernetes.",
    "Emily Davis is a research scientist at OpenAI. Email: emily.davis@openai.com. Phone: +1-212-555-3344. Location: San Francisco, CA.",
    "Michael Brown earned a Bachelor's degree in Data Science from the University of Toronto. He focuses on statistical modeling, SQL, and Python.",
    "Apple Inc., Cupertino, CA. Principal AI Engineer with 10 years of experience in speech recognition and large-scale ML systems. Skills: C++, PyTorch, ONNX.",
    "Daniel Kim is a full-stack developer at Shopify. Email: daniel.kim@shopify.com. Phone: +1-647-555-9981. Location: Toronto, ON.",
    "Sarah Wilson has a Master of Science in Software Engineering from Carnegie Mellon University. She specializes in distributed systems and cloud computing.",
    "IBM Research, Yorktown Heights, NY. Applied Scientist with 8 years of experience in natural language understanding and information extraction. Skills: Python, spaCy, Hugging Face."
]


def create_pdf_with_reportlab(text: str, output_path: str) -> None:
    """Create a PDF file from text using reportlab."""
    if not HAS_REPORTLAB:
        raise ImportError("reportlab is required for PDF generation")

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Create a paragraph with the text
    # Wrap text to fit page width
    para = Paragraph(text, styles['Normal'])
    story.append(para)

    doc.build(story)


def create_pdf_with_fpdf(text: str, output_path: str) -> None:
    """Create a PDF file from text using fpdf (fallback)."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf is required. Install with: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add text with word wrapping
    pdf.multi_cell(0, 10, text)

    pdf.output(output_path)


def create_image(text: str, output_path: str, width: int = 1200, padding: int = 40) -> None:
    """Create a PNG image file from text using PIL."""
    if not HAS_PIL:
        raise ImportError("Pillow is required for image generation")

    # Try to use a nice font, fallback to default if not available
    font_size = 24
    try:
        # Try common system fonts (Windows)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Try macOS/Linux fonts
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                try:
                    # Try Linux alternative
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Calculate text dimensions with proper word wrapping
    # Create a temporary image and draw object for text measurement
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    words = text.split()
    lines = []
    current_line = []
    max_line_width = width - 2 * padding

    for word in words:
        # Test if adding this word would exceed the line width
        test_line = ' '.join(current_line + [word])
        bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= max_line_width or not current_line:
            # Word fits or it's the first word on a new line
            current_line.append(word)
        else:
            # Word doesn't fit, start a new line
            lines.append(' '.join(current_line))
            current_line = [word]

    # Add the last line
    if current_line:
        lines.append(' '.join(current_line))

    # Calculate image height using actual text measurements
    line_height = font_size + 10  # Add some spacing
    total_height = len(lines) * line_height + 2 * padding

    # Create image with white background
    img = Image.new('RGB', (width, total_height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw text lines
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill='black', font=font)
        y += line_height

    # Save image
    img.save(output_path, 'PNG')


def generate_files(texts: List[str], output_dir: str = ".", prefix: str = "test_resume_ner") -> None:
    """
    Generate PDF and image files for each text line.

    Args:
        texts: List of text strings to convert
        output_dir: Directory to save files
        prefix: Prefix for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating files in: {output_path.absolute()}")

    for idx, text in enumerate(texts, start=1):
        # Generate PDF
        pdf_path = output_path / f"{prefix}_{idx}.pdf"
        try:
            if HAS_REPORTLAB:
                create_pdf_with_reportlab(text, str(pdf_path))
                print(f"✓ Created: {pdf_path.name}")
            else:
                create_pdf_with_fpdf(text, str(pdf_path))
                print(f"✓ Created: {pdf_path.name}")
        except Exception as e:
            print(f"✗ Failed to create PDF {pdf_path.name}: {e}")

        # Generate image
        img_path = output_path / f"{prefix}_{idx}.png"
        try:
            create_image(text, str(img_path))
            print(f"✓ Created: {img_path.name}")
        except Exception as e:
            print(f"✗ Failed to create image {img_path.name}: {e}")

    print(f"\nCompleted! Generated {len(texts)} pairs of PDF and image files.")


if __name__ == "__main__":
    # Generate files in the current directory
    generate_files(TEST_TEXTS, output_dir=".", prefix="test_resume_ner")
