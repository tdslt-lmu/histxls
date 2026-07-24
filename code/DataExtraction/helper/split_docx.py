"""
Split a DOCX file into multiple parts, either at a page break or at a specific string.

Configure INPUT_FILE and either SPLIT_AT_PAGE or SPLIT_AT_STRING below.
SPLIT_AT_STRING accepts a list of strings for multiple splits, e.g. ["350.", "527."].
"""

import os
from docx import Document
from docx.shared import Pt
from typing import Tuple, List


# ===== CONFIGURATION =====
INPUT_FILE = "data/FII_1464-1721.docx"

# Choose ONE splitting method:
# Method 1: Split by page number (uses page breaks)
SPLIT_AT_PAGE = None  # Set to page number (e.g., 85) or None to use string search

# Method 2: Split by finding specific strings (can be a list for multiple splits)
SPLIT_AT_STRING = ["1600."]  # List of strings to search for (e.g., ["350.", "527.", "600."])
                            # If 1 string: creates 2 parts
                            # If 3 strings: creates 4 parts (beginning → string1 → string2 → string3 → end)
# =========================


def copy_paragraph_with_formatting(source_para, target_doc):
    target_para = target_doc.add_paragraph()
    
    # Copy paragraph format
    target_para.paragraph_format.alignment = source_para.paragraph_format.alignment
    target_para.paragraph_format.left_indent = source_para.paragraph_format.left_indent
    target_para.paragraph_format.right_indent = source_para.paragraph_format.right_indent
    target_para.paragraph_format.first_line_indent = source_para.paragraph_format.first_line_indent
    target_para.paragraph_format.space_before = source_para.paragraph_format.space_before
    target_para.paragraph_format.space_after = source_para.paragraph_format.space_after
    target_para.style = source_para.style
    
    # Copy runs with formatting
    for run in source_para.runs:
        target_run = target_para.add_run(run.text)
        target_run.bold = run.bold
        target_run.italic = run.italic
        target_run.underline = run.underline
        target_run.font.size = run.font.size
        target_run.font.name = run.font.name
    
    return target_para


def count_page_breaks(doc: Document, up_to_paragraph: int = None) -> int:
    page_break_count = 0
    end_index = up_to_paragraph if up_to_paragraph is not None else len(doc.paragraphs)
    
    for i, paragraph in enumerate(doc.paragraphs[:end_index]):
        # Check if paragraph contains a page break
        if '\f' in paragraph.text or '\x0c' in paragraph.text:
            page_break_count += 1
            continue
        
        for run in paragraph.runs:
            if '\f' in run.text or '\x0c' in run.text:
                page_break_count += 1
                break
            
            # Check XML for page break elements
            if run._element.xml:
                if 'w:br' in run._element.xml and 'w:type="page"' in run._element.xml:
                    page_break_count += 1
                    break
    
    return page_break_count


def split_docx_at_string(input_path: str, search_string: str) -> Tuple[str, str]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading document: {input_path}")
    doc = Document(input_path)
    
    print(f"Document has {len(doc.paragraphs)} paragraphs")
    print(f"Searching for string: '{search_string}'")
    
    # Find the paragraph index containing the search string
    split_index = None
    
    for i, paragraph in enumerate(doc.paragraphs):
        # Check if paragraph text matches the search string (stripped of whitespace)
        if paragraph.text.strip() == search_string.strip():
            split_index = i
            print(f"✓ Found exact match at paragraph {i}: '{paragraph.text.strip()}'")
            break
    
    if split_index is None:
        # Trying a less strict match...
        print(f"No exact match found. Trying partial match...")
        for i, paragraph in enumerate(doc.paragraphs):
            if search_string.strip() in paragraph.text.strip():
                split_index = i
                print(f"Found partial match at paragraph {i}: '{paragraph.text.strip()}'")
                break
    
    if split_index is None:
        raise ValueError(f"Could not find string '{search_string}' in document")
    
    # Create output filenames
    base_name = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1]
    output_file1 = f"{base_name}_part1{ext}"
    output_file2 = f"{base_name}_part2{ext}"
    
    # Create first document (paragraphs 0 to split_index-1)
    print(f"Creating first document: {output_file1}")
    doc1 = Document()
    
    # Copy styles from original document
    doc1.styles._element = doc.styles._element
    
    for i in range(split_index):
        # Copy paragraph with all formatting
        source_para = doc.paragraphs[i]
        target_para = doc1.add_paragraph()
        
        # Copy paragraph format
        target_para.paragraph_format.alignment = source_para.paragraph_format.alignment
        target_para.paragraph_format.left_indent = source_para.paragraph_format.left_indent
        target_para.paragraph_format.right_indent = source_para.paragraph_format.right_indent
        target_para.paragraph_format.first_line_indent = source_para.paragraph_format.first_line_indent
        target_para.paragraph_format.space_before = source_para.paragraph_format.space_before
        target_para.paragraph_format.space_after = source_para.paragraph_format.space_after
        target_para.style = source_para.style
        
        # Copy runs with formatting
        for run in source_para.runs:
            target_run = target_para.add_run(run.text)
            target_run.bold = run.bold
            target_run.italic = run.italic
            target_run.underline = run.underline
            target_run.font.size = run.font.size
            target_run.font.name = run.font.name
    
    doc1.save(output_file1)
    print(f"Saved {output_file1} ({split_index} paragraphs)")
    
    # Create second document (paragraphs from split_index to end)
    print(f"Creating second document: {output_file2}")
    doc2 = Document()
    
    # Copy styles from original document
    doc2.styles._element = doc.styles._element
    
    for i in range(split_index, len(doc.paragraphs)):
        source_para = doc.paragraphs[i]
        target_para = doc2.add_paragraph()
        
        target_para.paragraph_format.alignment = source_para.paragraph_format.alignment
        target_para.paragraph_format.left_indent = source_para.paragraph_format.left_indent
        target_para.paragraph_format.right_indent = source_para.paragraph_format.right_indent
        target_para.paragraph_format.first_line_indent = source_para.paragraph_format.first_line_indent
        target_para.paragraph_format.space_before = source_para.paragraph_format.space_before
        target_para.paragraph_format.space_after = source_para.paragraph_format.space_after
        target_para.style = source_para.style
        
        for run in source_para.runs:
            target_run = target_para.add_run(run.text)
            target_run.bold = run.bold
            target_run.italic = run.italic
            target_run.underline = run.underline
            target_run.font.size = run.font.size
            target_run.font.name = run.font.name
    
    doc2.save(output_file2)
    print(f"Saved {output_file2} ({len(doc.paragraphs) - split_index} paragraphs)")
    
    return output_file1, output_file2


def split_docx_at_strings(input_path: str, search_strings: List[str]) -> List[str]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading document: {input_path}")
    doc = Document(input_path)
    
    print(f"Document has {len(doc.paragraphs)} paragraphs")
    print(f"Searching for {len(search_strings)} split point(s): {search_strings}")
    
    # Find all paragraph indices containing the search strings (in order)
    split_indices = []
    last_found_index = -1
    
    for search_string in search_strings:
        found_index = None
        print(f"\nSearching for: '{search_string}'")
        
        # Search from the last found index onwards
        for i in range(last_found_index + 1, len(doc.paragraphs)):
            paragraph = doc.paragraphs[i]
            
            # Check if paragraph text matches the search string (stripped of whitespace)
            if paragraph.text.strip() == search_string.strip():
                found_index = i
                print(f"Found exact match at paragraph {i}: '{paragraph.text.strip()}'")
                break
        
        if found_index is None:
            print(f"No exact match found. Trying partial match...")
            for i in range(last_found_index + 1, len(doc.paragraphs)):
                paragraph = doc.paragraphs[i]
                if search_string.strip() in paragraph.text.strip():
                    found_index = i
                    print(f"Found partial match at paragraph {i}: '{paragraph.text.strip()}'")
                    break
        
        
        split_indices.append(found_index)
        last_found_index = found_index
    
    # Create output filenames
    base_name = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1]
    output_files = []
    
    # Create all parts
    # Part 1: from beginning to first split
    # Part 2: from first split to second split
    # ...
    # Part N+1: from last split to end
    
    num_parts = len(split_indices) + 1
    
    for part_num in range(num_parts):
        output_file = f"{base_name}_part{part_num + 1}{ext}"
        output_files.append(output_file)
        
        # Determine paragraph range for this part
        if part_num == 0:
            # First part: beginning to first split
            start_idx = 0
            end_idx = split_indices[0]
        elif part_num == num_parts - 1:
            # Last part: last split to end
            start_idx = split_indices[-1]
            end_idx = len(doc.paragraphs)
        else:
            # Middle parts: from split N to split N+1
            start_idx = split_indices[part_num - 1]
            end_idx = split_indices[part_num]
        
        print(f"\nCreating part {part_num + 1}: {output_file}")
        
        # Create new document
        new_doc = Document()
        new_doc.styles._element = doc.styles._element
        
        # Copy paragraphs in range
        for i in range(start_idx, end_idx):
            copy_paragraph_with_formatting(doc.paragraphs[i], new_doc)
        
        new_doc.save(output_file)
        print(f"Saved {output_file}")
    
    return output_files


def split_docx_at_page(input_path: str, split_page: int) -> Tuple[str, str]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    doc = Document(input_path)
    
    # Find the paragraph index where we should split
    target_page_breaks = split_page - 1
    current_page = 1
    split_index = None
    
    for i, paragraph in enumerate(doc.paragraphs):
        # Count page breaks up to current paragraph
        page_breaks_before = count_page_breaks(doc, i)
        current_page = page_breaks_before + 1
        
        if page_breaks_before >= target_page_breaks and split_index is None:
            split_index = i
            print(f"Split point found at paragraph {i} (page {current_page})")
            break
    
    if split_index is None:
        total_pages = count_page_breaks(doc, len(doc.paragraphs)) + 1
        raise ValueError(f"Could not find split point at page {split_page}. Document has approximately {total_pages} pages.")
    
    # Create output filenames
    base_name = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1]
    output_file1 = f"{base_name}_part1{ext}"
    output_file2 = f"{base_name}_part2{ext}"
    
    # Create first document (paragraphs 0 to split_index-1)
    print(f"Creating first document: {output_file1}")
    doc1 = Document()
    
    # Copy styles from original document
    doc1.styles._element = doc.styles._element
    
    for i in range(split_index):
        # Copy paragraph with all formatting
        source_para = doc.paragraphs[i]
        target_para = doc1.add_paragraph()
        
        # Copy paragraph format
        target_para.paragraph_format.alignment = source_para.paragraph_format.alignment
        target_para.paragraph_format.left_indent = source_para.paragraph_format.left_indent
        target_para.paragraph_format.right_indent = source_para.paragraph_format.right_indent
        target_para.paragraph_format.first_line_indent = source_para.paragraph_format.first_line_indent
        target_para.paragraph_format.space_before = source_para.paragraph_format.space_before
        target_para.paragraph_format.space_after = source_para.paragraph_format.space_after
        target_para.style = source_para.style
        
        # Copy runs with formatting
        for run in source_para.runs:
            target_run = target_para.add_run(run.text)
            target_run.bold = run.bold
            target_run.italic = run.italic
            target_run.underline = run.underline
            target_run.font.size = run.font.size
            target_run.font.name = run.font.name
    
    doc1.save(output_file1)
    print(f"Saved {output_file1} ({split_index} paragraphs)")
    
    # Create second document (paragraphs from split_index to end)
    print(f"Creating second document: {output_file2}")
    doc2 = Document()
    
    # Copy styles from original document
    doc2.styles._element = doc.styles._element
    
    for i in range(split_index, len(doc.paragraphs)):
        # Copy paragraph with all formatting
        source_para = doc.paragraphs[i]
        target_para = doc2.add_paragraph()
        
        # Copy paragraph format
        target_para.paragraph_format.alignment = source_para.paragraph_format.alignment
        target_para.paragraph_format.left_indent = source_para.paragraph_format.left_indent
        target_para.paragraph_format.right_indent = source_para.paragraph_format.right_indent
        target_para.paragraph_format.first_line_indent = source_para.paragraph_format.first_line_indent
        target_para.paragraph_format.space_before = source_para.paragraph_format.space_before
        target_para.paragraph_format.space_after = source_para.paragraph_format.space_after
        target_para.style = source_para.style
        
        # Copy runs with formatting
        for run in source_para.runs:
            target_run = target_para.add_run(run.text)
            target_run.bold = run.bold
            target_run.italic = run.italic
            target_run.underline = run.underline
            target_run.font.size = run.font.size
            target_run.font.name = run.font.name
    
    doc2.save(output_file2)
    print(f"Saved {output_file2} ({len(doc.paragraphs) - split_index} paragraphs)")
    
    return output_file1, output_file2


def main():
    print(f"Input file: {INPUT_FILE}")
    
    # Determine which splitting method to use
    if SPLIT_AT_PAGE is not None:
        print(f"Split method: By page number")
        print(f"Split at page: {SPLIT_AT_PAGE}")
        print("="*60)
        print()
        
        try:
            output1, output2 = split_docx_at_page(INPUT_FILE, SPLIT_AT_PAGE)
            print()
            print("="*60)
            print("✓ Split completed successfully!")
            print(f"  Part 1: {output1}")
            print(f"  Part 2: {output2}")
            print("="*60)
        except Exception as e:
            print()
            print("="*60)
            print(f"✗ Error: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
    
    elif SPLIT_AT_STRING is not None:
        print(f"Split method: By search string(s)")
        
        # Handle both single string (backward compatibility) and list
        if isinstance(SPLIT_AT_STRING, str):
            search_strings = [SPLIT_AT_STRING]
        else:
            search_strings = SPLIT_AT_STRING
        
        print(f"Search string(s): {search_strings}")
        print(f"Will create {len(search_strings) + 1} parts")
        print("="*60)
        print()
        
        try:
            output_files = split_docx_at_strings(INPUT_FILE, search_strings)
            print()
            print("="*60)
            print("✓ Split completed successfully!")
            for i, output_file in enumerate(output_files):
                print(f"  Part {i + 1}: {output_file}")
            print("="*60)
        except Exception as e:
            print()
            print("="*60)
            print(f"✗ Error: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
    
    else:
        print()
        print("="*60)
        print("✗ Error: Please configure either SPLIT_AT_PAGE or SPLIT_AT_STRING")
        print("="*60)


if __name__ == "__main__":
    main()
