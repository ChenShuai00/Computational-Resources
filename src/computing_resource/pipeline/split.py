from pypdf import PdfReader, PdfWriter
import os
import argparse

def extract_and_remove_last_two_pages(input_pdf_path, checklist_output_path):
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)

    if total_pages < 2:
        raise ValueError(f"PDF {input_pdf_path} has fewer than 2 pages; cannot extract the last two pages.")

    # --- Extract the last two pages into the checklist file. ---
    checklist_writer = PdfWriter()
    for i in range(total_pages - 2, total_pages):
        checklist_writer.add_page(reader.pages[i])

    with open(checklist_output_path, "wb") as f:
        checklist_writer.write(f)

    # --- Keep the first N-2 pages and overwrite the original file. ---
    main_writer = PdfWriter()
    for i in range(total_pages - 2):
        main_writer.add_page(reader.pages[i])

    # Overwrite the original file. Use with care.
    with open(input_pdf_path, "wb") as f:
        main_writer.write(f)

def main():
    parser = argparse.ArgumentParser(description="Split the last two paper pages into a checklist PDF")
    parser.add_argument("--input-dir", default="paper_pdf/acl2023")
    parser.add_argument("--output-dir", default="checklist_pdf/acl2023")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for input_file in os.listdir(args.input_dir):
        if not input_file.lower().endswith(".pdf"):
            continue
        input_file_path = os.path.join(args.input_dir, input_file)
        output_file_path = os.path.join(
            args.output_dir,
            input_file.replace(".pdf", ".checklist.pdf"),
        )
        extract_and_remove_last_two_pages(input_file_path, output_file_path)
        print(f"Processed {input_file_path}: saved the last two pages to {output_file_path} and removed them from the original file.")


if __name__ == "__main__":
    main()
