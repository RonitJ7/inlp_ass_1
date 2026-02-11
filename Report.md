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

## Task 1.2

### Whitespace
 The whitespace tokenizer is pretty simple. We look through the line and idenfity if the character is a "word character", which means it checks if it belongs to Letters, Marks or Numbers unicode categories. If no, it is treated as a separate token (unless it is a space obviously). We split if the current character is a space , but don't include it as a token.

### Regex

### BPE
We begin by replacing space by the end of word(EOW) token (/w). 
For BPE :
1. We first keep track of all the words in the entire corpus. This number comes up to around 700,000 for both languges. We make a counter called word_freq to count (self explanatory I know , but still) how many times a word occurs in the corpus. Then we convert all the words to list form (with each letter being an entry in the list) ending with EOW.

2. The naive BPE would proceed as follows: We would first calculate pair wise counts for each pair. Then we would choose the one with the largest count and merge. Then we again calculate pair wise counts for each pair and so on. This however would have a time complexity of O(number of merge operations * number of words) which would be 50,000 * 700,000, which is FAR TOO MUCH. 

3. So we do the following fairly simple optimization: we only calculate pair wise counts for each pair once. After we choose the one with the largest count and merge , we only change the words which have the pair we are merging. This optimization improves speed to where I can run it and it takes a reasonable time. 

4. Further optimizations are possible, but I feel the tradeoff of complexity needed to time saved is not worth it. 

5. For encoding, the naive way is doing the merges in the order we performed them while training. But then for this , again for each token , we will have 10k merge operations, which is not computationally feasible.

6. So instead , we again do a very simple optimization. For each word, we first split it into a list (eg. List becomes ['L','i','s','t']). Then we consider what merges we can do and perform the one with highest priority. We keep doing this until we can't anymore. Also, if a word is already used, just cache it and directly used the cached word. 

7. A few more implementation details: 
    1. word_freq is a counter that keeps track of words and their frequencies. We use this to create word_symbols and word_freqs which respectively store the words' proper list representation appended with a EOW and their frequencies.
    2. We then use pair_counts to calculate freq of all the pair counts. We only do this in full once. We also use pair_to_words to keep track of which word (specifically w_id) contains the pair. 
    3. Then we get into the merge loop. We first get the best pair  while ensuring we aren't merging EOW or punctuation. We temporarily subtract all the contributions of these words to the frequencies of the pairs. Then we apply the merge in words actually containing the pair. Then we readd the contributions. This ensures that counting is done correctly.  This concludes the training. 
    4. For tokenization , implementation is super simple so there's no need to go over it.  
    
## Task 1.3


# Task 2
Other than the one mentioned in the problem statement PDF, there is another assumption we implicitly make when using 4 - grams : we consider that the previous 4 tokens contain all the relevant information for the current token. Mathematically we make the Markov assumption where : P(t_n | t_1 , t_2 ... t_(n-1)) = P(t_n | t_(n-1),t_(n-2),t_(n-3)) , where t_i is the ith token.

## Task 2.1

1. Training: To create a 4-gram model, we need to know how many times a particular tuple of length 1-4 has appeared in the sequence. We calculate these using dictionaries (unigram, trigram etc). Now , for a tuple (a,b,c,d) we also need to know how many times d has appeared after the tuple (a,b,c). We calculate these using dictionaries again (prev3, prev2 etc) . 
2. Prediction: Given an incomplete sentence we have to keep completing the sentence. It's an autocomplete problem. To simulate this, we feed the first few tokens in the test set to the model. Then we keep taking the most probable token to complete the sentence. How do we find the most probable token? We see the past 3 tokens, iterate along prev3 and see which has highest value. If we have never seen these 3 tokens before, we "backoff" to looking at only 2 of the prev tokens (prev2) and so on.  

## Task 2.2

### No Smoothing with backoff
This is similar to prediction, but we just calculate probability of the correct target word.  Fairly self explanatory. But one important point to note is that if there are no instances in the training set even for unigrams, we will just put probability as 1e-18 to avoid explosion to infinity while still seeing huge perplexity values as expected whitout smoothing. This shows us the difference in tokenizers while still giving us much higher values compared to without smoothing. 

### Kneser - Key
The intuition for kneser key smoothing can be explained with the popular Francisco example, where even though the frequency of the word might be high, but it only appears after San, so it is not a good word to use when autocompleting. So we balance it out by also considering how many different tokens appear before the current one. 
The formula:
P(w,h) = (max(c(w|h)- D, 0)) / c(h) + beta * P(w,h_(i-1))
where beta = (discount * t_h)/c(h)
and t_h = number of unique continuation tokens after the context
Here, we need a basecase for when i = 0 i.e there is no context. For this case, probability of a word is how many times it appears after a prev token upon number of two token pairs. 

As can be seen, the computation for P(w,context) depends on P(w,context[1:]), so we use a recursive structure to handle this. You can see this structure in the code. We use nexttoken to compute the values that we need.
The function compute_counts_kneser_key computes what we need for the basecase P(w). Otherwise we use nexttoken dicts calculated during the training. 

### Witten - Bell
The intuition for Witten Bell is that sequences that have lots of different tokens following them are more diverse, so we won't be surprised if there's a new word there (for example say "She is"). In contrast to this there are certain contexts which rarely appear with new words (San in San Francisco). To account for this Witten Bell gives some distribution (proportional to the amount of types that appear after a context) to such contexts.
Concretely, the formulas for probability:
Let N = Number of continuations to the context
T = Types of Tokens in continuation to the context
p(w,context) = \lambda * p_ml + (1-\lambda)*p_continuation
where p_ml = c(w,context)/c(context)
p_continuation = p(w, context[1: ])
lambda = N/(N+T)

The base case (i.e when context length is 0), is:
P(c) = c(c)/c(all tokens)

This can be fairly easily done using self.nexttoken  dicitonaries built during training and a recursive structure. 

## Task 2.3
Perplexity calculation: Perplexity is the "braching factor". It can be seen as the number of tokens that the model uniformly chooses between every step. The definition is (/pi (P(w_1,w_2 .. w_n)))^(-1/N). It can equivalently calculated as exp(-nll) over the sequence. We first do \sumlog(p(w_i | w_i-1,w_i-2,w_i-3)). we then do exp(-nll/n) to get perplexity. We return the average perplexity over all tokens.

## Task 2.4

(This is meant to be a table, pls render it as such)
Results table with epsilon = 1e-100
Tokenizer:           BPE    WS      Regex
Smoothing Method:
None                15088052286.34     8770450589463055.00   45313024577344576.00
Kneser-Key          280.07              2071.69                 2682.35
Witten Bell         305.94              2304.56                 3007.43

Results table with epsilon = 1e-12
Tokenizer:           BPE    WS      Regex
Smoothing Method:
None                395.46      1755.54     2230.58
Kneser-Key          280.07      530.34      611.93        
Witten Bell         305.94      589.95      686.09

These are the results for perplexities.

