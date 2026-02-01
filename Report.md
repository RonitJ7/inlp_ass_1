# Task 1
## Task 1.1
I first looked up common methods to clean large datasets for NLP tasks. I found a standard procedure called NFC normalization done on these datasets. I implemented this along with suggestions given in the assignment doc. 

The general motivation is that we want the text to be as close to normal language as possible without having random characters that can confuse the model.

Various cleaning done:
1. Unicode normalization: Unicode has various ways to write the same text. There are some characters that can be represented in multiple ways. We obviously want to avoid this because it can lead to incorrect counts for n-grams, and generally incorrect representations for various tokens. So , what NFC normalization does is convert these different equivalent representations into a single one. 

Code:
line = unicodedata.normalize("NFC", line)

2. Removing common invisible characters: There are various common invisible characters that are often found in text. The full list can be found in the code. These characters are simply removed from the text so that they don't become separate bogus tokens. 

Code:
line = line.replace(ch, "")

3. Removing control/format characters: These are characters with unicode format CC and CF. Control characters could be stuff like NUL characters , ESC character and format characters is stuff like word joiner etc. We remove these. 
Code:
s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})

4. Then we normalize whitespaces (i.e if there are multiple whitespaces or a tab character etc , then we will just convert all that to a single space.)

Code:
s = re.sub(r"\s+", " ", s).strip()