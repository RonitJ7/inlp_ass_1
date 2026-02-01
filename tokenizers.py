import json
from utils import parse_jsonl

def clean_and_split_dataset(path,text_field,train_ratio = 0.8,val_ratio = 0.1):

    lang_lines = parse_jsonl(path=path,text_field=text_field)
    total_lines = len(lang_lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)
    lang_train = lang_lines[:train_end]
    lang_val = lang_lines[train_end:val_end]
    lang_test = lang_lines[val_end:]
    return lang_train, lang_val, lang_test


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
