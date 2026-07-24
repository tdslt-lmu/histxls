# Structured Extraction of Medieval Charters

This folder is about the following Chapter of my Thesis: 3 - Dataset.

It contains the scripts and data for the automated extraction of charter attributes (e.g. German summary, Latin text) from the charters of Frederick II. The source documents are available as DOCX and HTML files and are transformed into structured CSV files.

Two approaches were implemented and compared:

## Approaches

### Regex Extraction
Rule-based extraction using regular expressions and positional logic directly from the DOCX files.  
Script and documentation: [`RegexExtraction/`](RegexExtraction/)

### LLM Extraction
AI-based extraction using the Gemini API from HTML exports of the source documents.  
Script and documentation: [`LLMExtraction/`](LLMExtraction/)

## Project Structure

| Directory | Contents |
|---|---|
| `data/` | Source documents (DOCX and HTML exports) |
| `csv/regex-extract/` | Results of the regex extraction |
| `csv/llm-extract/` | Results of the LLM extraction |
| `helper/` | Utility scripts (splitting DOCX files, printing specific charters) |
| `logs/` | Log files from the LLM extraction |

## Setup

> **Note:** This code can be executed on any machine — no GPU or cluster access required.

Python 3.9+, dependencies via Pipenv:

```bash
pipenv install
pipenv shell
```

## Usage

To run either approach, simply execute the `extract.py` file in the corresponding folder:

```bash
# Regex Extraction
python RegexExtraction/extract.py

# LLM Extraction
python LLMExtraction/extract.py
```

> **Important:** For the LLM Extraction, you need to fill in the `GEMINI_API_KEYS` list at the top of `LLMExtraction/extract.py` with your API keys before getting started.
