import os
from utils import parse_jsonl
import unicodedata
from collections import Counter,defaultdict
from tqdm import tqdm
import json

def save_tokenized_jsonl(path, tokenized_sents, field="tokens"):
    # create parent dir if needed
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sent in tokenized_sents:
            f.write(json.dumps({field: sent}, ensure_ascii=False) + "\n")

def load_tokenized_jsonl(path, field="tokens"):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj[field])
    return out

def clean_and_split_dataset(path,text_field,train_ratio = 0.8,val_ratio = 0.1):
    
    lang_lines = parse_jsonl(path=path,text_field=text_field)
    total_lines = len(lang_lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)
    lang_train = lang_lines[:train_end]
    lang_val = lang_lines[train_end:val_end]
    lang_test = lang_lines[val_end:]

    return lang_train, lang_val, lang_test

def _is_word_char(ch: str) -> bool:
    # Letters (L*), Marks (M*), Numbers (N*)
    return unicodedata.category(ch)[0] in {"L", "M", "N"}

def whitespace_tokenizer(lines):
    """Tokenizes lines into tokens based on whitespace and character types."""
    tokens = []
    for line in lines:
        buf = []
        sent = []
        for ch in line:
            if ch.isspace():
                if buf:
                    sent.append("".join(buf))
                    buf = []
            elif _is_word_char(ch):
                buf.append(ch)
            else:
                if buf:
                    sent.append("".join(buf))
                    buf = []
                sent.append(ch)
        if buf:
            sent.append("".join(buf))
        
        tokens.append(sent)

    return tokens


def regex_tokenizer(lines):
    pass

class BPETokenizer:
    def __init__(self, merge_operations, end_of_word="</w>", min_pair_freq=2):
        self.merge_operations = merge_operations
        self.end_of_word = end_of_word
        self.min_pair_freq = min_pair_freq
        self.merges = []  # learned merge rules in order
        self.merge_ranks = {} # merge pair -> rank in list
        self.encoder_cache = {} # word -> corresponding bpe encoded representation

    def train(self, lines):
        """
        Incremental BPE training:
        - Build word frequencies
        - Initialize each word as characters + </w>
        - Maintain pair_counts and pair_to_words so each merge only touches affected words
        - Early stop when best pair frequency < min_pair_freq
        """
        self.merges = []

        # 1) Count word types (training data only)
        word_freq = Counter()
        for line in lines:
            for word in line.split():
                if word:
                    word_freq[word] += 1

        # 2) Store words as symbol lists + frequencies (indexed by word_id)
        word_symbols = []
        word_freqs = []
        for w, freq in word_freq.items():
            word_symbols.append(list(w) + [self.end_of_word])
            word_freqs.append(freq)

        # 3) Initialize pair counts and inverted index: pair -> set(word_id)
        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for wid, symbols in enumerate(word_symbols):
            freq = word_freqs[wid]
            for p in self._pairs_in(symbols):
                pair_counts[p] += freq
                pair_to_words[p].add(wid)

        # 4) Merge loop (incremental updates)
        for _ in tqdm(range(self.merge_operations)):
            best_pair, best_count = self._best_pair(pair_counts)
            if best_pair is None or best_count < self.min_pair_freq:
                break
            
            affected = pair_to_words.get(best_pair)
            if not affected:
                # stale entry safeguard
                pair_counts.pop(best_pair, None)
                continue

            affected = list(affected)

            # Apply merge only to words that contain best_pair
            for wid in affected:
                old = word_symbols[wid]
                freq = word_freqs[wid]

                # Remove this word's current pair contributions
                for p in self._pairs_in(old):
                    pair_counts[p] -= freq
                    pair_to_words[p].discard(wid)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)

                # Merge in the word
                new = self._merge_symbols_once(old, best_pair)
                word_symbols[wid] = new

                # Add updated pair contributions
                for p in self._pairs_in(new):
                    pair_counts[p] += freq
                    pair_to_words[p].add(wid)

            # This pair should no longer exist anywhere after merging all affected words
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)

    def encode_word(self, word):
        symbols = list(word) + [self.end_of_word]
        cached = self.encoder_cache.get(word)
        if( cached is not None):
            return cached
        
        while True:
            best_rank = self.merge_operations + 1
            best_pair = None
            for p in self._pairs_in(symbols):
                if p in self.merge_ranks:
                    rank = self.merge_ranks[p]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = p
            if best_rank == self.merge_operations + 1:
                break
            symbols = self._merge_symbols_once(symbols, best_pair)
        
        if symbols and symbols[-1] == self.end_of_word:
            symbols = symbols[:-1]
        self.encoder_cache[word] = symbols
        return symbols

    def encode_line(self, line):
        out = []
        for w in line.split():
            out.extend(self.encode_word(w))
        return out

    @staticmethod
    def _pairs_in(symbols):
        # Adjacent pairs in a single word's symbol sequence
        for i in range(len(symbols) - 1):
            yield (symbols[i], symbols[i + 1])

    @staticmethod
    def _has_punct_or_symbol(sym):
        # True if ANY char in the symbol is punctuation (P*) or symbol (S*)
        for ch in sym:
            if unicodedata.category(ch)[0] in {"P", "S"}:
                return True
        return False
    
    def _best_pair(self, pair_counts):
        if not pair_counts:
            return None,0
        best_pair = 0
        best_count = -1
        for (a,b), count in pair_counts.items():
            # Skip anything involving end-of-word
            if a == self.end_of_word or b == self.end_of_word:
                continue

            # Skip anything involving punctuation/symbols
            if self._has_punct_or_symbol(a) or self._has_punct_or_symbol(b):
                continue

            if count > best_count:
                best_count = count
                best_pair = (a,b)
        return best_pair, best_count
    
    @staticmethod
    def _merge_symbols_once(symbols, pair):
        """Merge all occurrences of `pair` in `symbols` in a single pass."""
        a, b = pair
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

if __name__ == "__main__":
    #Prepare datasets
    path = "data/cc100_en.jsonl"
    text_field = "text"
    eng_train, eng_val, eng_test = clean_and_split_dataset(path,text_field)
    print(f"English Train Size: {len(eng_train)}")
    print(f"English Validation Size: {len(eng_val)}")
    print(f"English Test Size: {len(eng_test)}")
    path = "data/cc100_mn.jsonl"
    text_field = "text"
    mn_train, mn_val, mn_test = clean_and_split_dataset(path,text_field)
    print(f"Mongolian Train Size: {len(mn_train)}")
    print(f"Mongolian Validation Size: {len(mn_val)}")
    print(f"Mongolian Test Size: {len(mn_test)}")

    for lang in ["eng","mn"]:
        print(f"Training BPE tokenizer for {lang}...")
        bpe_tokenizer = BPETokenizer(merge_operations=10000)
        bpe_tokenizer.train(eval(f"{lang}_train"))
        for line in eval(f"{lang}_train")[:5]:
            print("Original Line:", line)
            print("BPE Encoded:", bpe_tokenizer.encode_line(line))
        for s in ["train","val","test"]:
            print(f"Tokenizing {lang} {s} set with BPE...")
            tok_path = f"cache/{lang}_{s}_bpe.jsonl"
            lines = eval(f"{lang}_{s}")
            if os.path.exists(tok_path):
                lines_bpe = load_tokenized_jsonl(tok_path)
            else:
                lines_bpe = [bpe_tokenizer.encode_line(line) for line in tqdm(lines)]
                save_tokenized_jsonl(tok_path, lines_bpe)

