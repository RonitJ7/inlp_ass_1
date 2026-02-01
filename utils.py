import unicodedata
import re

_COMMON_INVISIBLES = {
    "\ufeff",  # BOM / zero width no-break space
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
}

def clean_line(line):    

    #Unicode normalization
    s = unicodedata.normalize("NFC", line)

    #Removing common invisible characters
    for ch in _COMMON_INVISIBLES:
        s = s.replace(ch, "")

    #Remove ocntrol and format characters
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})

    #Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s

def clean_corpus(lines, min_chars):
    cleaned_lines = []
    for line in enumerate(lines):
        s = clean_line(line)
        if len(s) < min_chars:
            continue
        cleaned_lines.append(s)
    
    return cleaned_lines
