# Generate Test Files for Resume NER Testing

This script generates PDF and PNG image files from text lines used for Resume NER testing.

## Installation

Install the required dependencies:

```bash
pip install -r requirements_generate.txt
```

Or install individually:
```bash
pip install reportlab Pillow fpdf2
```

## Usage

Run the script:

```bash
python generate_test_files.py
```

This will generate:
- `test_resume_ner_1.pdf` through `test_resume_ner_10.pdf`
- `test_resume_ner_1.png` through `test_resume_ner_10.png`

All files will be created in the current directory.

## Customization

You can modify the script to:
- Change the output directory by editing the `output_dir` parameter
- Change the filename prefix by editing the `prefix` parameter
- Add or modify test texts in the `TEST_TEXTS` list

## Notes

- The script uses `reportlab` for PDF generation (with `fpdf2` as fallback)
- The script uses `Pillow` (PIL) for image generation
- If libraries are missing, the script will print warnings but continue with available libraries
