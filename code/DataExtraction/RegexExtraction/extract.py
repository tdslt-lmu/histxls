import re
import csv
import os
from docx import Document

# List of all relevant files to process
FILES = [
    "FII_1-170",
    "FII_171-235", 
    "FII_236-299",
    "FII_300-363",
    "FII_364-426",
    "FII_427-657",
    "FII_658-929",
    "FII_930-1199",
    "FII_1200-1463",
    "FII_1464-1721"
]

DEBUG_FILE = None
DEBUG_URKUNDE = None

def is_urkunden_nummer(text):
    """Check if a line is a charter number (e.g. '1464.' or '1516. zweifelhaft')"""
    return re.match(r'^\d{1,4}\.\s*.*$', text.strip())

def is_ort_datum(text):
    """Check if a line contains a location and date"""
    # Various formats:
    # "Ravenna, 1232 Januar"
    # "Ravenna, (1232) Januar 11"
    # "Ravenna, 1232 März (nach 24)"
    return re.match(r'^\s*[A-Za-z]+,\s*\(?\d{3,4}\)?\s+[A-Za-z]+(\s+\d{1,2}|\s*\([^)]+\))?\s*$', text)

def find_multiline_paragraph_start(lines, start_idx, end_idx):
    """Find the first multi-line block in the given range.
    If none exists, fall back to the first very long single-line block (>500 chars).
    """
    # Pass 1: Look for multi-line blocks only and return the first one
    i = start_idx
    while i < end_idx:
        if lines[i].strip() == "":
            i += 1
            continue
        block_start = i
        line_count = 0
        while i < end_idx and lines[i].strip() != "":
            line_count += 1
            i += 1
        if line_count > 1:
            return block_start
        if i < end_idx and lines[i].strip() == "":
            i += 1

    # Pass 2: No multi-line block found → look for a very long single-liner
    i = start_idx
    while i < end_idx:
        if lines[i].strip() == "":
            i += 1
            continue
        block_start = i
        total_chars = 0
        # One block until next empty line (will be a single line at this point)
        while i < end_idx and lines[i].strip() != "":
            total_chars += len(lines[i].strip())
            i += 1
        if total_chars > 500:
            return block_start
        if i < end_idx and lines[i].strip() == "":
            i += 1

    return -1

def extract_urkunde(urkunde_lines, debug=False):
    """Extract a single charter from the given lines"""
    if not urkunde_lines:
        return None
    
    urkunde = {}
    i = 0
    
    # 1. Charter number (first line)
    if not is_urkunden_nummer(urkunde_lines[0]):
        if debug:
            print(f"  ✗ First line is not a charter number: '{urkunde_lines[0]}'")
        return None
    
    # Extract only the numeric part (without qualifiers like "zweifelhaft")
    nummer_match = re.match(r'^(\d{1,4})\.', urkunde_lines[0].strip())
    urkunde['nummer'] = nummer_match.group(1) if nummer_match else urkunde_lines[0].strip('.')
    if debug:
        print(f"  Nummer: {urkunde['nummer']}")
    
    # 2. German summary - always the first line after the charter number
    i = 1
    if i < len(urkunde_lines):
        urkunde['summary_de'] = urkunde_lines[i].strip()
        if debug:
            print(f"  Summary: {urkunde['summary_de'][:100]}...")
        i += 1
    else:
        urkunde['summary_de'] = ""
        if debug:
            print("  ✗ No summary found!")
        return None
    
    # 3. Location & date - always the second line after the charter number
    if i < len(urkunde_lines):
        urkunde['ort_datum'] = urkunde_lines[i].strip()
        start_anchor = i
        if debug:
            print(f"  Location/Date: '{urkunde['ort_datum']}' (line {start_anchor})")
        i += 1
    else:
        urkunde['ort_datum'] = ""
        if debug:
            print("  ✗ No location/date found!")
        return None
    
    # 4. Find first multi-line paragraph (end anchor)
    end_anchor = find_multiline_paragraph_start(urkunde_lines, start_anchor + 1, len(urkunde_lines))
    if debug:
        print(f"  Multi-line paragraph starts at line: {end_anchor}")
    
    # 5. Meta info blocks between start_anchor and end_anchor
    meta_blocks = []
    j = start_anchor + 1
    
    while j < end_anchor and j < len(urkunde_lines):
        # Skip empty lines
        if urkunde_lines[j].strip() == "":
            j += 1
            continue
        
        # Collect block until next empty line
        block_lines = []
        while j < end_anchor and j < len(urkunde_lines) and urkunde_lines[j].strip() != "":
            block_lines.append(urkunde_lines[j])
            j += 1
        
        if block_lines:
            meta_text = " ".join(block_lines).strip()
            meta_blocks.append(meta_text)
            if debug:
                print(f"    Meta info {len(meta_blocks)}: {meta_text[:50]}...")
    
    # Assign meta info fields
    urkunde['meta_info_1'] = meta_blocks[0] if len(meta_blocks) > 0 else ""
    urkunde['meta_info_2'] = meta_blocks[1] if len(meta_blocks) > 1 else ""
    urkunde['meta_info_3'] = meta_blocks[2] if len(meta_blocks) > 2 else ""
    urkunde['meta_info_4'] = meta_blocks[3] if len(meta_blocks) > 3 else ""
    
    # 6. multi-line paragraphs...
    if end_anchor >= 0:
        zustand_lines = []
        j = end_anchor
        
        # Collect all lines of the multi-line paragraph
        while j < len(urkunde_lines) and urkunde_lines[j].strip() != "":
            zustand_lines.append(urkunde_lines[j])
            j += 1
        
        urkunde['zustand_schreibstil'] = " ".join(zustand_lines).strip()
        if debug:
            print(f"  Condition: {urkunde['zustand_schreibstil'][:50]}...")
        
        # Skip empty line after condition/writing style
        if j < len(urkunde_lines) and urkunde_lines[j].strip() == "":
            j += 1
    else:
        urkunde['zustand_schreibstil'] = ""
        j = len(urkunde_lines)
    
    # 7. Latin text (remaining lines)
    latin_lines = []
    verweis = ""
    
    while j < len(urkunde_lines):
        line = urkunde_lines[j]
        
        # Check if the line is a standalone reference (SP. X) or (B. X)
        if re.match(r'^\((SP|B)\.\s*[^)]+\)$', line.strip()):
            verweis = line.strip()
            break
        
        # Check if a reference appears at the end of the line
        verweis_match = re.search(r'(\((SP|B)\.\s*[^)]+\))$', line)
        if verweis_match:
            # Split the line
            text_before_verweis = line[:verweis_match.start(1)].strip()
            if text_before_verweis:
                latin_lines.append(text_before_verweis)
            verweis = verweis_match.group(1)
            break
        
        # Regular Latin text line
        if line.strip():
            latin_lines.append(line)
        j += 1
    
    urkunde['latin_text'] = " ".join(latin_lines).strip()
    urkunde['verweis'] = verweis
    
    if debug:
        print(f"  Latin: {urkunde['latin_text'][:50]}...")
        print(f"  Verweis: {urkunde['verweis']}")
    
    return urkunde

def process_file(file_name):
    """Process a single file"""
    is_debug_file = (DEBUG_FILE is None or file_name == DEBUG_FILE)
    
    if is_debug_file:
        print(f"\n{'='*60}")
        print(f"Processing file: {file_name}")
        if DEBUG_FILE == file_name and DEBUG_URKUNDE:
            print(f"DEBUG MODE: Looking for charter {DEBUG_URKUNDE}")
        print(f"{'='*60}")
    
    # Check for .doc or .docx
    input_path = f"data/{file_name}.docx"
    if not os.path.exists(input_path):
        input_path = f"data/{file_name}.doc"
        if not os.path.exists(input_path):
            if is_debug_file:
                print(f"ERROR: File {file_name} not found (.doc or .docx)")
            return None
    
    output_path = f"csv/{file_name}.csv"
    
    # Load document
    try:
        doc = Document(input_path)
        if is_debug_file:
            print(f"File loaded: {input_path}")
    except Exception as e:
        if is_debug_file:
            print(f"ERROR loading {input_path}: {e}")
        return None

    # Extract all lines (including empty lines!)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        lines.append(text)  # Keep empty lines too!

    if is_debug_file:
        print(f"Total lines found: {len(lines)}")

    # Main logic: split text into individual charters
    if is_debug_file:
        print("Splitting text into charters...")

    # Find all charter numbers
    urkunden_indices = []
    debug_urkunde_found = False
    debug_urkunde_idx = -1
    
    for i, line in enumerate(lines):
        if is_urkunden_nummer(line):
            urkunden_indices.append(i)
            # Check if this is the target debug charter
            if DEBUG_URKUNDE and file_name == DEBUG_FILE:
                nummer_match = re.match(r'^(\d{1,4})\.', line.strip())
                nummer = nummer_match.group(1) if nummer_match else line.strip('.')
                if nummer == DEBUG_URKUNDE:
                    debug_urkunde_found = True
                    debug_urkunde_idx = len(urkunden_indices) - 1
                    print(f"DEBUG: Charter {DEBUG_URKUNDE} found at line {i} (index {debug_urkunde_idx})")

    if is_debug_file:
        print(f"Charter numbers found: {len(urkunden_indices)}")
        
    # Special debug info for target charter
    if DEBUG_URKUNDE and file_name == DEBUG_FILE and not debug_urkunde_found:
        print(f"DEBUG: Charter {DEBUG_URKUNDE} NOT found!")
        print("Charters found in this file:")
        for i, idx in enumerate(urkunden_indices[:10]):  # Show first 10
            line = lines[idx]
            nummer_match = re.match(r'^(\d{1,4})\.', line.strip())
            nummer = nummer_match.group(1) if nummer_match else line.strip('.')
            print(f"  {i}: Charter {nummer} at line {idx}")
        if len(urkunden_indices) > 10:
            print(f"  ... and {len(urkunden_indices) - 10} more")

    # Extract each charter
    records = []
    for i in range(len(urkunden_indices)):
        start_idx = urkunden_indices[i]
        end_idx = urkunden_indices[i + 1] if i + 1 < len(urkunden_indices) else len(lines)
        
        # Determine if this is the debug charter
        is_debug_urkunde = (i == debug_urkunde_idx and DEBUG_URKUNDE and file_name == DEBUG_FILE)
        
        # Logging
        if is_debug_urkunde:
            print(f"\nDEBUG: Processing charter {DEBUG_URKUNDE} (lines {start_idx}-{end_idx-1})")
            print(f"Lines for charter {DEBUG_URKUNDE}:")
            for j in range(start_idx, min(start_idx + 15, end_idx)):  # Show first 15 lines
                print(f"  {j}: '{lines[j]}'")
            if end_idx - start_idx > 15:
                print(f"  ... and {end_idx - start_idx - 15} more lines")
        elif is_debug_file and (i < 5 or i % 50 == 0):  # First 5 and every 50th
            print(f"Processing charter {i + 1}: {lines[start_idx].strip()}")
        
        # Extract lines for this charter
        urkunde_lines = lines[start_idx:end_idx]
        
        # Extract charter with debug flag
        urkunde = extract_urkunde(urkunde_lines, debug=is_debug_urkunde)
        
        if urkunde:
            records.append(urkunde)
            if is_debug_urkunde:
                print(f"DEBUG: Charter {DEBUG_URKUNDE} extracted successfully!")
        else:
            if is_debug_urkunde:
                print(f"DEBUG: ERROR extracting charter {DEBUG_URKUNDE}!")
            elif is_debug_file and i < 5:
                print(f"  x Extraction failed for {lines[start_idx].strip()}")

    if is_debug_file:
        print(f"Total charters extracted: {len(records)}")

    # Save CSV
    fieldnames = [
        'nummer',
        'summary_de', 
        'ort_datum',
        'meta_info_1',
        'meta_info_2', 
        'meta_info_3',
        'meta_info_4',
        'zustand_schreibstil',
        'latin_text',
        'verweis'
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            # Fill missing fields with empty string
            for field in fieldnames:
                if field not in record:
                    record[field] = ""
            writer.writerow(record)

    if is_debug_file:
        print(f"✅ Done! {len(records)} charters saved to {output_path}")

        # Brief stats
        print(f"\nStats for {file_name}:")
        print(f"- Charters with German summary: {sum(1 for r in records if r['summary_de'])}")
        print(f"- Charters with Latin text: {sum(1 for r in records if r['latin_text'])}")
        print(f"- Charters with location/date: {sum(1 for r in records if r['ort_datum'])}")
        print(f"- Charters with reference: {sum(1 for r in records if r['verweis'])}")
        print(f"- Charters with condition/style: {sum(1 for r in records if r['zustand_schreibstil'])}")

    return len(records)

# Main loop: process all files
if __name__ == "__main__":
    if DEBUG_FILE and DEBUG_URKUNDE:
        print(f"DEBUG MODE: Focusing on {DEBUG_FILE}, charter {DEBUG_URKUNDE}")
    else:
        print("Starting batch processing of all files...")
    
    total_urkunden = 0
    successful_files = 0
    
    # If debug mode is active, process only the debug file
    files_to_process = [DEBUG_FILE] if DEBUG_FILE else FILES
    
    for file_name in files_to_process:
        try:
            result = process_file(file_name)
            if result is not None:
                total_urkunden += result
                successful_files += 1
        except Exception as e:
            print(f"ERROR in {file_name}: {e}")
    
    if DEBUG_FILE and DEBUG_URKUNDE:
        print(f"\nDEBUG ANALYSIS COMPLETE")
    else:
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Files processed successfully: {successful_files}/{len(files_to_process)}")
        print(f"Total charters extracted: {total_urkunden}")
        print(f"CSV files saved to 'csv/' folder")
