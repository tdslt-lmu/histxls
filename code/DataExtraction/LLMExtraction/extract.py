import os
import csv
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import google.generativeai as genai

# Its important here to define the class names of the charter splitter html element!
HTML_FILES: List[tuple] = [
    (os.path.join("data", "html", "FII_1-170.html"), "c25"),
    (os.path.join("data", "html", "FII_171-235.html"), "c20"),
    (os.path.join("data", "html", "FII_236-299.html"), "c7"),
    (os.path.join("data", "html", "FII_300-363.html"), "c16"),
    (os.path.join("data", "html", "FII_364-426.html"), "c22"),
    (os.path.join("data", "html", "FII_427-526.html"), "c5"),
    (os.path.join("data", "html", "FII_527-657.html"), "c14"),
    (os.path.join("data", "html", "FII_658-799.html"), "c8"),
    (os.path.join("data", "html", "FII_800-929.html"), "c14"),
    (os.path.join("data", "html", "FII_930-1049.html"), "c9"),
    (os.path.join("data", "html", "FII_1050-1199.html"), "c11"),
    (os.path.join("data", "html", "FII_1200-1329.html"), "c10"),
    (os.path.join("data", "html", "FII_1330-1463.html"), "c5"),
    (os.path.join("data", "html", "FII_1464-1599.html"), "c12"),
    (os.path.join("data", "html", "FII_1600-1721.html"), ("c13", "c14")),
]


TARGET_NUMMERS: List[str] = [
    # Fehlende Urkunden
    # "46", "188", "283", "308", "334", "361", "363", "379", "393", "420",
    # "424", "439", "508", "528", "544", "661", "665", "685", "705", "727",
    # "734", "766", "770", "788", "790", "799", "801", "802", "850", "868",
    # "880", "881", "914", "1045", "1068", "1075", "1149", "1150", "1152", "1167",
    # "1200", "1201", "1212", "1265", "1271", "1276", "1277", "1282", "1284", "1286",
    # "1293", "1317", "1318", "1319", "1320", "1323", "1331", "1338", "1341", "1349",
    # "1350", "1351", "1352", "1353", "1354", "1359", "1361", "1363", "1364", "1367",
    # "1369", "1370", "1371", "1372", "1385", "1407", "1414", "1415", "1445", "1450",
    # "1457", "1491", "1493", "1494", "1500", "1501", "1505", "1506", "1508", "1509",
    # "1511", "1512", "1513", "1514", "1515", "1521", "1522", "1549", "1551", "1552",
    # "1553", "1554", "1580", "1636", "1637", "1639", "1641", "1656", "1657", "1659",
    # "1664", "1668", "1695", "1716", "1717",
]

GEMINI_API_KEYS: List[str] = [
# Add your Gemini API keys here
]

# CONFIG: Gemini model
GEMINI_MODEL = "gemini-flash-latest"
# ALT_GEMINI_MODEL = "gemini-2.5-pro"
ALT_GEMINI_MODEL = "gemini-flash-latest"

# Global variable to track current API key index
_current_api_key_index = 0
_api_keys_exhausted = set()  # Track which keys have hit quota

# Console color config
USE_COLOR = True

# ANSI colors
COLOR_NUMMER = "\033[96m"
COLOR_SUCCESS = "\033[92m"
COLOR_ERROR = "\033[91m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"


def get_current_api_key() -> str:
    global _current_api_key_index
    
    if not GEMINI_API_KEYS:
        raise ValueError("No API keys configured! Please add your API keys to GEMINI_API_KEYS list.")
    
    # Check if all keys are exhausted
    if len(_api_keys_exhausted) >= len(GEMINI_API_KEYS):
        raise RuntimeError("All API keys have hit their quota limits!")
    
    # Skip exhausted keys
    while _current_api_key_index in _api_keys_exhausted:
        _current_api_key_index = (_current_api_key_index + 1) % len(GEMINI_API_KEYS)
    
    return GEMINI_API_KEYS[_current_api_key_index]


def rotate_api_key():
    global _current_api_key_index
    
    # Mark current key as exhausted
    _api_keys_exhausted.add(_current_api_key_index)
    print_error(f"API key #{_current_api_key_index + 1} has hit quota limit")
    
    # Find next available key
    original_index = _current_api_key_index
    _current_api_key_index = (_current_api_key_index + 1) % len(GEMINI_API_KEYS)
    
    # Skip already exhausted keys
    attempts = 0
    while _current_api_key_index in _api_keys_exhausted and attempts < len(GEMINI_API_KEYS):
        _current_api_key_index = (_current_api_key_index + 1) % len(GEMINI_API_KEYS)
        attempts += 1
    
    if _current_api_key_index in _api_keys_exhausted:
        raise RuntimeError("All API keys have hit their quota limits!")
    
    print_success(f"Switching to API key #{_current_api_key_index + 1}")
    
    # Reconfigure genai with new key
    genai.configure(api_key=get_current_api_key())


def is_quota_error(error: Exception) -> bool:
    error_str = str(error).lower()
    quota_indicators = [
        'quota',
        'rate limit',
        'resource exhausted',
        '429',
        'too many requests',
        'resource_exhausted'
    ]
    return any(indicator in error_str for indicator in quota_indicators)


def is_urkunden_nummer(text: str) -> bool:
    """
    Check if a line is a charter number.
    Valid formats:
    - "3."
    - "1464."
    - "3. Fälschung"
    - "1516. Verfälscht"
    - "222. Zweifelhaft"
    Number must be between 1 and 3000.
    """
    stripped = text.strip()
    # Remove HTML entities like &nbsp;
    stripped = stripped.replace('\u00a0', ' ').replace('&nbsp;', ' ').strip()
    
    # Match: number + dot + optional (Fälschung or Verfälscht)
    match = re.match(r'^(\d{1,4})\.\s*(Fälschung|Verfälscht|Zweifelhaft)?\s*$', stripped, re.IGNORECASE)
    if match:
        nummer = int(match.group(1))
        return 1 <= nummer <= 3000
    return False


def normalize_nummer(value: str) -> str:
    """Normalize charter number"""
    if value is None:
        return ""
    v = str(value).strip()
    # Remove HTML entities
    v = v.replace('\u00a0', ' ').replace('&nbsp;', ' ').strip()
    if v.endswith('.'):
        v = v[:-1]
    return v


def extract_superscript_classes(soup) -> List[str]:
    """
    Extract CSS class names that have 'vertical-align: super' from the HTML.
    Returns a list of class names.
    """
    superscript_classes = []
    
    # Find all <style> tags
    style_tags = soup.find_all('style')
    
    for style_tag in style_tags:
        if style_tag.string:
            css_content = style_tag.string
            
            # Find all CSS rules with vertical-align: super
            # Pattern matches: .classname { ... vertical-align: super; ... }
            lines = css_content.split('\n')
            current_class = None
            in_rule = False
            has_super = False
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts with a class selector (e.g., ".c3 {")
                if line.startswith('.') and '{' in line:
                    # If we were tracking a previous rule, reset
                    current_class = None
                    in_rule = False
                    has_super = False
                    
                    # Extract class name between . and {
                    class_match = re.match(r'\.([a-zA-Z0-9_-]+)\s*\{', line)
                    if class_match:
                        current_class = class_match.group(1)
                        in_rule = True
                
                # Check if this rule has vertical-align: super
                if in_rule and 'vertical-align:' in line and 'super' in line:
                    has_super = True
                
                # End of rule - check if we should add this class
                if in_rule and '}' in line:
                    if current_class and has_super:
                        superscript_classes.append(current_class)
                    current_class = None
                    in_rule = False
                    has_super = False
    
    return superscript_classes


def remove_superscript_notes(element, superscript_classes: List[str]):
    """
    Remove all superscript notes from an element:
    1. All <sup> elements (including their children)
    2. All spans with classes that use vertical-align: super
    This modifies the element in place.
    """
    if element is None:
        return
    
    # Remove all <sup> elements (including their children)
    for sup in element.find_all('sup'):
        sup.decompose()
    
    # Remove all spans with superscript classes
    if superscript_classes:
        for span in element.find_all('span', class_=superscript_classes):
            span.decompose()


def has_enumeration_pattern(text: str) -> bool:
    """
    Check if text contains enumeration patterns like 'a)', '1)', 'b)', etc.
    Returns True if the pattern is found.
    
    Only matches if the pattern appears at the start of the text or after whitespace,
    to avoid false positives from Latin words.
    """
    if not text:
        return False
    
    # Pattern matches: letter + ), number + ), or similar patterns
    # Must be at start or after whitespace to avoid matching within words
    # Examples: "a) text", " b) more", "1) item", " 2) next"
    # Will NOT match: "ad)" within a word, "domini D_" etc.
    pattern = r'(?:^|\s)[a-z]{1,2}\)|(?:^|\s)\d{1,2}\)'
    return bool(re.search(pattern, text.strip(), re.IGNORECASE))


def remove_enumeration_paragraphs(element):
    """
    Remove <p> tags that contain multiple spans with enumeration patterns.
    These are typically annotation paragraphs like:
    "2. a) domini D_ b) fehlt D_D__ c) Cristi E d) Federicus..."
    
    This modifies the element in place.
    """
    if element is None:
        return
    
    for p_tag in element.find_all('p'):
        spans = p_tag.find_all('span')
        
        enum_count = 0
        enum_texts = []
        for span in spans:
            span_text = span.get_text()
            if has_enumeration_pattern(span_text):
                enum_count += 1
                enum_texts.append(span_text[:20])  # Store first 20 chars for debugging
        
        # If more than one span has enumeration, remove the entire <p> tag
        if enum_count > 1:
            p_tag.decompose()


def extract_text_from_element(element, superscript_classes: List[str] = None, preserve_paragraphs: bool = False) -> str:
    if element is None:
        return ""
    
    if superscript_classes is None:
        superscript_classes = []
    
    # Convert element to string and replace &nbsp; with regular space
    # This must be done BEFORE parsing to ensure proper space handling
    html_str = str(element)
    html_str = html_str.replace('&nbsp;', ' ')
    
    # Make a copy with the replaced spaces
    element_copy = BeautifulSoup(html_str, 'html.parser')
    
    remove_superscript_notes(element_copy, superscript_classes)
    remove_enumeration_paragraphs(element_copy)
    
    if preserve_paragraphs:
        # Extract text with paragraph structure preserved
        result_parts = []
        processed_elements = set()
        
        # Recursively process element to extract text from block-level elements
        def process_element(elem):
            if elem in processed_elements:
                return
            
            if hasattr(elem, 'name') and elem.name in ['p', 'div', 'li', 'td', 'th', 'blockquote']:
                processed_elements.add(elem)
                
                elem_text = elem.get_text(separator='', strip=False)
                elem_text = elem_text.replace('\u00a0', ' ').replace('\xa0', ' ')
                elem_text = re.sub(r'\s+', ' ', elem_text).strip()
                
                # Only include non-empty content
                if elem_text:
                    result_parts.append(f"<text>{elem_text}</text>")
            
            # Recursively process children
            if hasattr(elem, 'children'):
                for child in elem.children:
                    if hasattr(child, 'name'):  # It's an element, not text
                        process_element(child)
        
        # Start processing from the root element
        process_element(element_copy)
        
        # If we didn't find any block elements, fall back to getting all text
        if not result_parts:
            all_text = element_copy.get_text(separator='', strip=False)
            all_text = all_text.replace('\u00a0', ' ').replace('\xa0', ' ')
            all_text = re.sub(r'\s+', ' ', all_text).strip()
            if all_text:
                result_parts.append(f"<text>{all_text}</text>")
        
        return '\n'.join(result_parts)
    else:
        # Get text without adding separators between elements
        # This preserves cases like "cere" + "a" = "cerea" (no space between spans)
        text = element_copy.get_text(separator='', strip=False)
        
        # Also replace unicode non-breaking space if any remain
        text = text.replace('\u00a0', ' ').replace('\xa0', ' ')
        
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


def split_into_charters(html_path: str, marker_class) -> List[Dict[str, str]]:
    """
    Split an HTML file into individual charters based on charter marker spans.
    Returns a list of dicts with 'nummer' and 'raw_text' keys.
    """
    if not os.path.exists(html_path):
        return []
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract superscript classes from CSS
    superscript_classes = extract_superscript_classes(soup)
    
    # Find all spans with the specified marker class(es) which mark charter numbers
    # Handle both single class string and tuple of classes
    if isinstance(marker_class, tuple):
        required_classes = set(marker_class)
        all_spans = soup.find_all('span')
        charter_markers = []
        
        for span in all_spans:
            span_classes = span.get('class', [])
            if isinstance(span_classes, str):
                span_classes = [span_classes]
            span_class_set = set(span_classes)
            
            if required_classes.issubset(span_class_set):
                charter_markers.append(span)
        
        print_progress(f"  Found {len(charter_markers)} spans with both classes")
    else:
        # Single class string
        charter_markers = soup.find_all('span', class_=marker_class)
    
    charters = []
    
    for i, marker in enumerate(charter_markers):
        # Extract the charter number from the marker's parent element
        marker_parent = marker.parent
        if marker_parent:
            marker_text = extract_text_from_element(marker_parent, superscript_classes)
        else:
            # Fallback to marker itself if no parent
            marker_text = extract_text_from_element(marker, superscript_classes)
        
        if not is_urkunden_nummer(marker_text):
            continue
        
        # Extract just the number
        nummer_match = re.match(r'^(\d{1,4})\.', marker_text.strip())
        if not nummer_match:
            continue
        
        nummer = nummer_match.group(1)
        
        # Collect all content until the next charter marker
        charter_parts = []
        current_element = marker.parent
        charter_parts.append(extract_text_from_element(current_element, superscript_classes, preserve_paragraphs=True))
        next_element = current_element.find_next_sibling()
        element_count = 0
        
        while next_element:
            element_count += 1
            # Only for debuggin: Show element type and classes
            elem_info = f"{next_element.name}"
            if hasattr(next_element, 'get') and next_element.get('class'):
                elem_info += f" class={' '.join(next_element.get('class'))}"
            
            # Need to check for both single class and tuple of classes
            next_marker = None
            if isinstance(marker_class, tuple):
                # Find span with all required classes
                required_classes = set(marker_class)
                for span in next_element.find_all('span'):
                    span_classes = span.get('class', [])
                    if isinstance(span_classes, str):
                        span_classes = [span_classes]
                    if required_classes.issubset(set(span_classes)):
                        next_marker = span
                        break
            else:
                # Single class
                next_marker = next_element.find('span', class_=marker_class)
            
            if next_marker:
                next_marker_text = extract_text_from_element(next_marker, superscript_classes, preserve_paragraphs=False)
                if is_urkunden_nummer(next_marker_text):
                    break
            
            # Add this element's text with paragraph structure preserved
            element_text = extract_text_from_element(next_element, superscript_classes, preserve_paragraphs=True)
            
            if element_text:
                charter_parts.append(element_text)
            
            next_element = next_element.find_next_sibling()
        
        # Combine all parts
        charter_text = '\n'.join(charter_parts)
        
        if charter_text.strip():
            charters.append({
                'nummer': nummer,
                'raw_text': charter_text,
                '_source': os.path.basename(html_path)
            })
    
    return charters


def extract_with_ai(raw_text: str, nummer: str, max_retries: int = 5, retry_delay: int = 30) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Send charter text to Gemini Flash 2.5 and extract structured data.
    """
    prompt = f"""You are a medieval charter transcription expert. You will receive a historical charter (Urkunde) text where each paragraph is wrapped in <text>...</text> tags.

The charter contains:
1. A German summary (Zusammenfassung) - typically one or more <text> paragraphs in German
2. A place and date line (e.g., "Palermo, (1199) Juni 22") - IMMEDIATELY after the German summary
3. Latin charter text - a SEQUENCE of consecutive <text> paragraphs containing ONLY Latin
4. Additional information (may contain mixed German/Latin)

Your task is to extract the german summary and the latin text and return them as JSON.

CRITICAL REQUIREMENTS:
- You MUST transcribe the text EXACTLY as it appears
- Do NOT change a single word
- Do NOT use synonyms or paraphrases
- Do NOT modernize spelling or grammar
- Keep ALL special characters, abbreviations, and formatting markers like [cite_start] or [cite: 540]
- Use the EXACT wording from the source text
- Remove the <text> tags from your output - only include the actual text content

IDENTIFYING THE LATIN TEXT:
- The Latin charter text consists of CONSECUTIVE <text> paragraphs that contain ONLY Latin words
- Look for a sequence where ALL paragraphs are purely Latin (no German mixed in)
- Individual <text> paragraphs that mix German and Latin (e.g., scholarly annotations with "vgl.", "siehe", etc.) are NOT part of the charter text
- The Latin charter typically starts with a formal opening like "IN NOMINE", "Fredericus", or similar
- Example pattern to look for:
  <text>IN NOMINE SANCTE ET INDIVIDUE TRINITATIS...</text>
  <text>Quapropter omnium tam presentium...</text>
  <text>Datum apud Palernum anno...</text>
  (all purely Latin, no German words)

SPECIAL CASE - Spaced Location Names:
Sometimes location names appear with spaces between each character, like "S. A n g e l o zu P r i z z i".
When you encounter such spaced-out names, you MUST normalize them by removing the extra spaces.
Examples:
- "S. A n g e l o zu P r i z z i" → "S. Angelo zu Prizzi"
- "S. M a r i a d e l l a S a m b u c i n a" → "S. Maria della Sambucina"
- "M o n t e f i a s c o n e" → "Montefiascone"
Apply this normalization to location names, monastery names, and place names in the German summary.

CRITICAL: Identifying WHERE the German summary ENDS:
The German summary ALWAYS ends IMMEDIATELY BEFORE the place/date line.
The place/date line typically appears in a format like:
- "Palermo, (1199) Juni 22"
- "Basel, 1212 September 26"
- "Rom, 1220 November 22"
This is a HARD BOUNDARY - the German summary STOPS here.

EVERYTHING that comes AFTER the place/date line is NOT part of the summary:
- Do NOT include the place/date line itself
- Do NOT include ANY text that appears after the place/date line
- Do NOT include archive information, manuscript details, literature references, or any other material after the date
- Only include the content description that appears BEFORE the place/date line

Example structure:
<text>Friedrich nimmt das Kloster unter seinen Schutz...</text>  ← THIS IS THE SUMMARY
<text>Palermo, (1199) Juni 22</text>  ← STOP HERE! This and everything after is NOT summary
<text>Original im Archiv...</text>  ← NOT part of summary
<text>Literatur: xyz...</text>  ← NOT part of summary

SPECIAL CASE - Charter Status Prefix:
Sometimes the charter text begins with a status marker in a separate <text> block, like:
- <text>2. Verfälscht</text>
- <text>171. Fälschung</text>
- <text>25. Zweifelhaft</text>
When you encounter such a prefix (number + dot + status word), you MUST OMIT this entire <text> block from the summary_de.
The summary should begin with the actual content description, not the status marker.
Example:
  Input: <text>2. Verfälscht</text>\n<text>Friedrich nimmt das Zisterzienserkloster S. Angelo zu Prizzi...</text>
  Output summary_de: "Friedrich nimmt das Zisterzienserkloster S. Angelo zu Prizzi..."

Return ONLY a valid JSON object with exactly these two keys:
{{
  "summary_de": "the complete German summary text WITHOUT the place/date and WITHOUT any charter status prefix, with normalized location names",
  "latin_text": "the complete Latin charter text (consecutive purely-Latin paragraphs only)"
}}

Charter number: {nummer}

Charter text:
{raw_text}

Remember: EXACT transcription is essential. Not a single word should be changed. Do NOT include the place/date line in summary_de. NORMALIZE spaced-out location names. Extract ONLY the consecutive purely-Latin paragraphs for latin_text."""

    for attempt in range(max_retries):
        try:
            # Use alternative model on 4th and 5th attempts (indices 3 and 4)
            current_model = ALT_GEMINI_MODEL if attempt >= 3 else GEMINI_MODEL
            if attempt >= 3:
                print_progress(f"  Switching to alternative model: {current_model}")
            
            model = genai.GenerativeModel(current_model)
            
            generation_config = {
                "temperature": 0.0,  # Deterministic output
                "top_p": 1.0,
                "top_k": 1,
                "max_output_tokens": 8192,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to find JSON in the response (in case it's wrapped in markdown)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required keys
            if 'summary_de' not in result or 'latin_text' not in result:
                raise ValueError(f"AI response missing required keys (has: {list(result.keys())})")
            
            # Success! Return the result with no error
            return (result, None)
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse AI response as JSON: {str(e)}"
            if attempt < max_retries - 1:
                print_error(f"Charter {nummer} - {error_msg}")
                print_progress(f"  Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print_error(f"Charter {nummer} - {error_msg} (all retries exhausted)")
                print_error(f"Response was: {response_text[:200]}...")
                return (None, error_msg)
        
        except Exception as e:
            error_msg = f"AI extraction failed: {str(e)}"
            
            # Check if this is a quota error
            if is_quota_error(e):
                print_error(f"Charter {nummer} - Quota limit hit!")
                try:
                    rotate_api_key()
                    # Retry immediately with new key (don't count as an attempt)
                    print_progress(f"  Retrying with new API key...")
                    continue
                except RuntimeError as rotate_error:
                    # All keys exhausted
                    return (None, str(rotate_error))
            
            # For non-quota errors, follow normal retry logic
            if attempt < max_retries - 1:
                print_error(f"Charter {nummer} - {error_msg}")
                print_progress(f"  Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print_error(f"Charter {nummer} - {error_msg} (all retries exhausted)")
                return (None, error_msg)
    
    # Should not reach here, but just in case
    return (None, "Unknown error: maximum retries exceeded")


def filter_charters(charters: List[Dict[str, str]], targets: List[str]) -> List[Dict[str, str]]:
    """Filter charters by target numbers"""
    if not targets:
        return charters
    target_set = {normalize_nummer(t) for t in targets}
    return [c for c in charters if normalize_nummer(c.get("nummer")) in target_set]


def print_success(msg: str):
    if USE_COLOR:
        print(f"{COLOR_SUCCESS}✓{COLOR_RESET} {msg}")
    else:
        print(f"✓ {msg}")


def print_error(msg: str):
    if USE_COLOR:
        print(f"{COLOR_ERROR}✗{COLOR_RESET} {msg}")
    else:
        print(f"✗ {msg}")


def print_progress(msg: str):
    if USE_COLOR:
        print(f"{COLOR_DIM}{msg}{COLOR_RESET}")
    else:
        print(msg)


def clean_text_for_csv(text: str) -> str:
    """
    Remove newline characters and normalize whitespace for CSV storage.
    Replaces newlines with spaces and collapses multiple spaces into one.
    """
    if not text:
        return ""
    # Replace all types of newlines with spaces
    cleaned = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Strip leading/trailing whitespace
    return cleaned.strip()


def write_log_file(log_path: str, html_file: str, successful_ids: List[str], 
                   failed_extractions: List[Dict[str, str]], total_charters: int, 
                   start_time: datetime, end_time: datetime):
    """
    Write a log file with extraction metadata.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    duration = (end_time - start_time).total_seconds()
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "html_file": html_file,
        "processing_time_seconds": round(duration, 2),
        "total_charters": total_charters,
        "successful_count": len(successful_ids),
        "failed_count": len(failed_extractions),
        "success_rate": f"{(len(successful_ids) / total_charters * 100) if total_charters > 0 else 0:.1f}%",
        "successful_ids": successful_ids,
        "failed_extractions": failed_extractions
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def write_output_csv(records: List[Dict[str, str]], out_path: str):
    """Write extracted records to CSV"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nummer", "summary_de", "latin_text"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "nummer": r.get("nummer", ""),
                "summary_de": clean_text_for_csv(r.get("summary_de", "")),
                "latin_text": clean_text_for_csv(r.get("latin_text", "")),
            })


def main():
    global _current_api_key_index, _api_keys_exhausted
    
    # Check for API keys
    if GEMINI_API_KEYS and GEMINI_API_KEYS[0] != "YOUR_API_KEY_1":
        # Use hardcoded keys
        print_success(f"Using {len(GEMINI_API_KEYS)} configured API key(s)")
        _current_api_key_index = 0
        _api_keys_exhausted = set()
        api_key = get_current_api_key()
        print_progress(f"Starting with API key #{_current_api_key_index + 1}")
    else:
        # Fall back to environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print_error("No API keys configured!")
            print("Please either:")
            print("  1. Set GEMINI_API_KEYS list in the script, OR")
            print("  2. Set GEMINI_API_KEY environment variable")
            return
        print_progress("Using API key from environment variable")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    print(f"Using Gemini model: {GEMINI_MODEL}\n")
    
    # Process each HTML file separately
    for html_path, marker_class in HTML_FILES:
        html_basename = os.path.basename(html_path)
        # Remove .html extension and add _html.csv
        csv_name = os.path.splitext(html_basename)[0] + "_html.csv"
        output_csv = os.path.join("csv", csv_name)
        
        print(f"\n{COLOR_SUCCESS}{'='*60}{COLOR_RESET}")
        print_progress(f"Processing {html_basename}...")
        print_progress(f"Output will be saved to: {csv_name}")
        print(f"{COLOR_SUCCESS}{'='*60}{COLOR_RESET}\n")
        
        # Read charters from this HTML file
        charters = split_into_charters(html_path, marker_class)
        print_progress(f"Found {len(charters)} charters in {html_basename}")
        
        # Filter by target numbers if specified
        selected = filter_charters(charters, TARGET_NUMMERS)
        print(f"Processing {len(selected)} charters\n")
        
        if len(selected) == 0:
            print_progress(f"No charters to process from {html_basename}, skipping...\n")
            continue
        
        # Track processing time and results
        start_time = datetime.now()
        successful_ids = []
        failed_extractions = []
        
        # Process each charter with AI
        extracted_records = []
        for i, charter in enumerate(selected, 1):
            nummer = charter['nummer']
            
            if USE_COLOR:
                print(f"\n{COLOR_DIM}[{i}/{len(selected)}]{COLOR_RESET} {COLOR_NUMMER}Charter {nummer}{COLOR_RESET} {COLOR_DIM}(from {html_basename}){COLOR_RESET}")
            else:
                print(f"\n[{i}/{len(selected)}] Charter {nummer} (from {html_basename})")
            
            # Extract with AI
            result, error = extract_with_ai(charter['raw_text'], nummer)
            
            if result:
                extracted_records.append({
                    'nummer': nummer,
                    'summary_de': result['summary_de'],
                    'latin_text': result['latin_text'],
                })
                successful_ids.append(nummer)
                print_success(f"Extracted charter {nummer}")
            else:
                failed_extractions.append({
                    'id': nummer,
                    'error': error or 'Unknown error'
                })
                print_error(f"Failed to extract charter {nummer}")
            
            # Sleep for 60 seconds between AI calls to avoid rate limits
            # if i < len(selected):  # Don't sleep after the last charter
                # print_progress(f"Waiting 60 seconds before next request...")
                # time.sleep(10)
        
        # Record end time
        end_time = datetime.now()
        
        # Sort by charter number
        def sort_key(r: Dict[str, str]):
            n = normalize_nummer(r.get("nummer", ""))
            try:
                return int(n)
            except ValueError:
                return 10**9
        extracted_records.sort(key=sort_key)
        
        # Write output CSV for this file
        write_output_csv(extracted_records, output_csv)
        
        # Write log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{os.path.splitext(html_basename)[0]}_{timestamp}.log.json"
        log_path = os.path.join("logs", log_filename)
        write_log_file(log_path, html_basename, successful_ids, failed_extractions, 
                      len(selected), start_time, end_time)
        
        print(f"\n{COLOR_SUCCESS}{'='*60}{COLOR_RESET}")
        print_success(f"Successfully extracted {len(extracted_records)}/{len(selected)} charters from {html_basename}")
        print_success(f"Saved to {output_csv}")
        print_success(f"Log file saved to {log_path}")
        print(f"{COLOR_SUCCESS}{'='*60}{COLOR_RESET}\n")
    
    print(f"\n{COLOR_SUCCESS}{'='*80}{COLOR_RESET}")
    print_success(f"All files processed!")
    
    # Report API key usage
    if len(GEMINI_API_KEYS) > 1 and GEMINI_API_KEYS[0] != "YOUR_API_KEY_1":
        print()
        print("API Key Usage:")
        if _api_keys_exhausted:
            print(f"  Keys that hit quota limits: {len(_api_keys_exhausted)}/{len(GEMINI_API_KEYS)}")
            exhausted_nums = [str(i+1) for i in sorted(_api_keys_exhausted)]
            print(f"  Exhausted keys: #{', #'.join(exhausted_nums)}")
        print(f"  Final active key: #{_current_api_key_index + 1}")
        print(f"  Available keys remaining: {len(GEMINI_API_KEYS) - len(_api_keys_exhausted)}/{len(GEMINI_API_KEYS)}")
    
    print(f"{COLOR_SUCCESS}{'='*80}{COLOR_RESET}\n")


if __name__ == "__main__":
    main()
