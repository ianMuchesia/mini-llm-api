import re

INPUT_FILE = "data/extracted/AA/wiki_00"   # adjust to your actual wikiextractor output path
OUTPUT_FILE = "data/swahili_clean.txt"
TARGET_BYTES = 10 * 1024 * 1024  # 1 MB


def clean_wikitext(text):
    # Remove wikitables entirely: {| ... |}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)

    # Remove infobox/template blocks: {{ ... }}
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)

    # Remove leftover infobox-style lines: |key = value or |- valign="top"
    text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\|-.*$', '', text, flags=re.MULTILINE)

    # Remove gallery blocks
    text = re.sub(r'<gallery>.*?</gallery>', '', text, flags=re.DOTALL)

    # Remove ref tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)

    # Remove remaining HTML tags EXCEPT <doc> tags (handled separately below)
    text = re.sub(r'<(?!/?doc\b)[^>]+>', '', text)

    # Replace common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&ndash;', '–')
    text = text.replace('&mdash;', '—')
    text = text.replace('&amp;', '&')

    # Remove Picha:/Category:/Jamii: lines
    text = re.sub(r'^\s*(Picha|Category|Jamii):.*$', '', text, flags=re.MULTILINE)

    # Remove redirect stubs
    text = re.sub(r'^#redirect.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Strip image/caption markup in any pipe order
    def strip_image_markup(line):
        if not re.search(r'\bthumb(nail)?\b|\d+px|alt=|\bframe\b|\bupright\b|\bleft\b|\bright\b|\bcenter\b', line):
            return line
        parts = line.split('|')
        keep_pattern = r'^\s*(thumbnail|thumb|frame|left|right|center|upright|\d+px|alt=.*)\s*$'
        kept = [p for p in parts if not re.match(keep_pattern, p, flags=re.IGNORECASE)]
        joined = '|'.join(kept).strip()
        return joined if joined and len(joined) > 3 else ''
    text = '\n'.join(strip_image_markup(line) for line in text.split('\n'))

    # Remove wiki links
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)

    # Remove bold/italic
    text = re.sub(r"'''", '', text)
    text = re.sub(r"''", '', text)

    # Headers
    text = re.sub(r'={2,}\s*([^=]+?)\s*={2,}', r'\1', text)

    # Bullets
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)

    # Drop boilerplate section markers
    boilerplate_lines = r'^\s*(Tanbihi|Tazama pia|Tazamia pia|Viungo vya nje|Marejeo)\s*$'
    text = re.sub(boilerplate_lines, '', text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(r'\[\[|\]\]|\[|\]', '', text)

    # Collapse blank lines and trailing whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def strip_doc_tags_and_dedupe_titles(text):
    # Remove <doc ...> opening tags and </doc> closing tags
    text = re.sub(r'<doc[^>]*>', '', text)
    text = re.sub(r'</doc>', '', text)

    # wikiextractor repeats the article title as the first line of the body
    lines = text.split('\n')
    cleaned_lines = []
    prev_blank = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        if prev_blank and stripped and i + 1 < len(lines):
            next_nonblank = None
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip():
                    next_nonblank = lines[j].strip()
                    break
            if next_nonblank == stripped:
                prev_blank = True
                continue
        cleaned_lines.append(line)
        prev_blank = (stripped == '')

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


# --- Run pipeline ---
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

stage1 = clean_wikitext(raw)
stage2 = strip_doc_tags_and_dedupe_titles(stage1)

# Trim to target size, cutting at a paragraph boundary
final = stage2
if len(final.encode('utf-8')) > TARGET_BYTES:
    truncated = final.encode('utf-8')[:TARGET_BYTES].decode('utf-8', errors='ignore')
    last_break = truncated.rfind('\n\n')
    if last_break > 0:
        truncated = truncated[:last_break]
    final = truncated

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final)

print(f"Original size: {len(raw)} chars")
print(f"After wikitext cleanup: {len(stage1)} chars")
print(f"After doc-tag/title dedup: {len(stage2)} chars")
print(f"Final (trimmed to ~{TARGET_BYTES // (1024*1024)}MB): {len(final)} chars ({len(final.encode('utf-8'))} bytes)")