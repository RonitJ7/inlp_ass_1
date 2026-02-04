import unicodedata
import re
import json
_COMMON_INVISIBLES = {
    "\ufeff",  # BOM / zero width no-break space
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
}

def load_tokenized_jsonl(path, field="tokens", max_lines=None):
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            row = row.strip()
            if not row:
                continue
            obj = json.loads(row)
            lines.append(obj[field])
    return lines

def parse_jsonl(path,text_field,encoding="utf-8",min_chars=8):
    raw_lines = []

    with open(path, "r", encoding=encoding,errors = "replace") as f: #Here, errors replace means invalid byte sequences will be replaced with a replacement unicode character : �, otherwise it will just crash if it sees a weird character which we don't want
        for line_no,line in enumerate(f,start = 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_no} due to JSON decode error: {e}")
                continue
            
            raw_lines.append(obj[text_field])

    return clean_corpus(raw_lines,min_chars=min_chars)

def clean_line(line):    

    #Unicode normalization
    s = unicodedata.normalize("NFC", line)

    #Removing common invisible characters
    for ch in _COMMON_INVISIBLES:
        s = s.replace(ch, "")

    #Remove control and format characters
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})

    #Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s

def clean_corpus(lines, min_chars):
    cleaned_lines = []
    for line in lines:
        s = clean_line(line)
        if len(s) < min_chars:
            continue
        cleaned_lines.append(s)
    
    return cleaned_lines
