from utils import load_tokenized_jsonl
from tqdm import tqdm 
import math
class LM:
    def __init__(self): 
        self.sequence_count4 = {}
        self.sequence_count3 = {}
        self.sequence_count2 = {}
        self.sequence_count1 = {}
        self.nexttoken3 = {}
        self.nexttoken2 = {}
        self.nexttoken1 = {}
        self.delims = {'.', '?', '!'}# Split sentences by '.' , '?' and '.'
        self.cache_kneser_key_prob = {}
        self.cache_witten_bell_prob = {}

    def split(self, line):
        curr = []
        sentences = []
        for tok in line:
            curr.append(tok)
            if tok in self.delims:
                if curr:
                    sentences.append(curr)
                    curr = []
        if curr:
            sentences.append(curr)
        return sentences
    
    def train(self, lines):
        print("Training LM...")
        print("Counting number of sentences...")
        sentences = []
        for line in tqdm(lines):
            #break into sentences
            sentences.extend(self.split(line))           

        print("Counting ngrams and next tokens... ")
        for sent in tqdm(sentences):
            sequence = ["<s>"]*3 + sent + ["</s>"]
            for i in range(3, len(sequence)):
                # Record counts of all n-grams up to 4-grams
                unigram = tuple(sequence[i:i+1])
                bigram = tuple(sequence[i-1:i+1])
                trigram = tuple(sequence[i-2:i+1])
                quadgram = tuple(sequence[i-3:i+1])
                if quadgram not in self.sequence_count4:
                    self.sequence_count4[quadgram] = 0
                self.sequence_count4[quadgram] += 1
                if trigram not in self.sequence_count3:
                    self.sequence_count3[trigram] = 0
                self.sequence_count3[trigram] += 1
                if bigram not in self.sequence_count2:
                    self.sequence_count2[bigram] = 0
                self.sequence_count2[bigram] += 1
                if unigram not in self.sequence_count1:
                    self.sequence_count1[unigram] = 0
                self.sequence_count1[unigram] += 1

                # Record counts of next tokens
                prev3 = tuple(sequence[i-3:i])
                prev2 = tuple(sequence[i-2:i])
                prev1 = tuple(sequence[i-1:i])
                if prev3 not in self.nexttoken3:
                    self.nexttoken3[prev3] = {}
                if sequence[i] not in self.nexttoken3[prev3]:
                    self.nexttoken3[prev3][sequence[i]] = 0
                self.nexttoken3[prev3][sequence[i]] += 1
                if prev2 not in self.nexttoken2:
                    self.nexttoken2[prev2] = {}
                if sequence[i] not in self.nexttoken2[prev2]:
                    self.nexttoken2[prev2][sequence[i]] = 0
                self.nexttoken2[prev2][sequence[i]] += 1
                if prev1 not in self.nexttoken1:
                    self.nexttoken1[prev1] = {}
                if sequence[i] not in self.nexttoken1[prev1]:
                    self.nexttoken1[prev1][sequence[i]] = 0
                self.nexttoken1[prev1][sequence[i]] += 1    
        
        self.tot_tokens = sum(self.sequence_count1.values())

    def predict(self,context):
        sentences = self.split(context)
        for sent in sentences:
            sentence = ["<s>"]*3 + sent
            for _ in range(50): #Generate atmost 50 tokens more
                prev3 = tuple(sentence[-3:])
                prev2 = tuple(sentence[-2:])
                prev1 = tuple(sentence[-1:])
                next_token = None

                # Try 4-gram
                if prev3 in self.nexttoken3:
                    next_tokens = self.nexttoken3[prev3]
                    next_token = max(next_tokens, key=next_tokens.get)

                # Backoff to 3-gram
                elif prev2 in self.nexttoken2:
                    next_tokens = self.nexttoken2[prev2]
                    next_token = max(next_tokens, key=next_tokens.get)

                # Backoff to 2-gram
                elif prev1 in self.nexttoken1:
                    next_tokens = self.nexttoken1[prev1]
                    next_token = max(next_tokens, key=next_tokens.get)

                # If no prediction possible, break
                if next_token is None or next_token == "</s>":
                    break

                sentence.append(next_token)
            # Print the generated sentence excluding the starting <s> tokens
            print(" ".join(sentence[3:]))

    def get_prob_no_smoothing(self, sentence):
        log_prob = 0.0

        for i in range(3, len(sentence)):
            w = sentence[i]
            prev3 = tuple(sentence[i-3:i])
            prev2 = tuple(sentence[i-2:i])
            prev1 = tuple(sentence[i-1:i])

            prob = None

            # 4-gram MLE from nexttoken3
            dist3 = self.nexttoken3.get(prev3)
            if dist3 and w in dist3:
                prob = dist3[w] / sum(dist3.values())

            else:
                # backoff to trigram (context len 2)
                dist2 = self.nexttoken2.get(prev2)
                if dist2 and w in dist2:
                    prob = dist2[w] / sum(dist2.values())
                else:
                    # backoff to bigram (context len 1)
                    dist1 = self.nexttoken1.get(prev1)
                    if dist1 and w in dist1:
                        prob = dist1[w] / sum(dist1.values())
                    else:
                        # no smoothing => zero prob => perplexity infinite
                        prob = 1e-12

            log_prob += math.log(prob)

        return log_prob
    
    def compute_counts_kneser_key(self):
        self.total_bigram_types = sum(len(next_tokens) for next_tokens in self.nexttoken1.values())
        self.left_context_count = {}
        for prev,dic in self.nexttoken1.items():
            for w in dic:
                if w not in self.left_context_count:
                    self.left_context_count[w] = 0
                self.left_context_count[w] += 1
           
    
    def kneser_key_prob(self, context, w, discount=0.75):
        prob = 0.0
        cached = self.cache_kneser_key_prob.get((context,w))
        if  cached is not None:
            return cached
        if len(context) == 3:
            if context not in self.nexttoken3:
                return self.kneser_key_prob(context[1:], w, discount)
            else:
                numerator = max(self.nexttoken3[context].get(w, 0) - discount, 0)
                denominator = sum(self.nexttoken3[context].values())
                prob = numerator / denominator
                t_h = len(self.nexttoken3[context])
                beta = (discount * t_h) / denominator 
                prob += beta*self.kneser_key_prob(context[1:], w, discount)

        elif len(context) == 2:
            if context not in self.nexttoken2:
                return self.kneser_key_prob(context[1:], w, discount)
            else:
                numerator = max(self.nexttoken2[context].get(w, 0) - discount, 0)
                denominator = sum(self.nexttoken2[context].values())
                prob = numerator / denominator
                t_h = len(self.nexttoken2[context])
                beta = (discount * t_h) / denominator 
                prob += beta*self.kneser_key_prob(context[1:], w, discount)
        elif len(context) == 1:
            if context not in self.nexttoken1:
                return self.kneser_key_prob(context[1:], w, discount)
            else:
                numerator = max(self.nexttoken1[context].get(w, 0) - discount, 0)
                denominator = sum(self.nexttoken1[context].values())
                prob = numerator / denominator
                t_h = len(self.nexttoken1[context])
                beta = (discount * t_h) / denominator 
                prob += beta*(self.left_context_count.get(w, 0) / self.total_bigram_types)

        elif len(context) == 0:
                prob = self.left_context_count.get(w, 0) / self.total_bigram_types

        self.cache_kneser_key_prob[context,w] = prob
        return prob
    
    def get_prob_kneser_key(self,sentence):
        log_prob = 0.0
        n_pred = 0
        for i in range(3, len(sentence)):
            context = tuple(sentence[i-3:i])
            w = sentence[i]
            prob = self.kneser_key_prob(context, w)
            if(prob == 0):
                prob = 1e-12
            log_prob += math.log(prob)
            n_pred += 1
        return log_prob
    
    def witten_bell_counts(self,context,w):
        prob = 0.0
        cached = self.cache_witten_bell_prob.get((context,w))
        if  cached is not None:
            return cached
        if len(context) == 3:
            if context not in self.nexttoken3:
                prob =  self.witten_bell_counts(context[1:], w)
            else:
                count_context = sum(self.nexttoken3[context].values())
                if count_context == 0:
                    return self.witten_bell_counts(context[1:], w)
                T = len(self.nexttoken3[context])
                lambda_ = count_context / (count_context + T)
                p_continuation = self.witten_bell_counts(context[1:], w)
                p_ml = self.nexttoken3[context].get(w, 0) / count_context
                prob += lambda_ * p_ml + (1 - lambda_) * p_continuation
        elif len(context) == 2:
            if context not in self.nexttoken2:
                prob = self.witten_bell_counts(context[1:], w)
            else:
                count_context = sum(self.nexttoken2[context].values())
                if count_context == 0:
                    return self.witten_bell_counts(context[1:], w)
                T = len(self.nexttoken2[context])
                lambda_ = count_context / (count_context + T)
                p_continuation = self.witten_bell_counts(context[1:], w)
                p_ml = self.nexttoken2[context].get(w, 0) / count_context
                prob += lambda_ * p_ml + (1 - lambda_) * p_continuation
        elif len(context) == 1:
            if context not in self.nexttoken1:
                prob =  self.witten_bell_counts(context[1:], w)
            else:
                count_context = sum(self.nexttoken1[context].values())
                if count_context == 0:
                    return self.witten_bell_counts(context[1:], w)
                T = len(self.nexttoken1[context])
                lambda_ = count_context / (count_context + T)
                p_continuation = self.witten_bell_counts(context[1:], w)
                p_ml = self.nexttoken1[context].get(w, 0) / count_context
                prob += lambda_ * p_ml + (1 - lambda_) * p_continuation
        elif len(context) == 0:
            prob = self.sequence_count1.get((w,), 0) / self.tot_tokens
        
        self.cache_witten_bell_prob[context,w] = prob
        return prob
    
    def get_witten_bell_prob(self, sentence):
        log_prob = 0.0
        for i in range(3, len(sentence)):
            context = tuple(sentence[i-3:i])
            w = sentence[i]
            prob = self.witten_bell_counts(context, w)
            if(prob == 0):
                prob = 1e-12
            log_prob += math.log(prob)
        return log_prob


if __name__ == "__main__":
    # Example usage
    corpus = load_tokenized_jsonl("cache/eng_train_bpe.jsonl")
    print(f"Loaded corpus with {len(corpus)} lines.")

    for tokenizer in ["bpe","rxt","ws"]:
        print("Training LM with", tokenizer, "tokenizer...")
        corpus = load_tokenized_jsonl(f"cache/eng_train_{tokenizer}.jsonl")
        lm = LM()
        lm.train(corpus)

        print(f"Evaluating perplexity for {tokenizer} tokenizer with no smoothing")
        test_corpus = load_tokenized_jsonl(f"cache/eng_test_{tokenizer}.jsonl")
        total_prob = 0.0
        count = 0
        for line in tqdm(test_corpus):
            no_tokens = len(line)
            sentence = ["<s>"]*3 + line + ["</s>"]
            prob = lm.get_prob_no_smoothing(sentence)
            total_prob += prob
            count += no_tokens+1

        corpus_perplexity = math.exp(-total_prob / max(count, 1))
        print(f"Average perplexity for {tokenizer}: {corpus_perplexity:.2f}")

        total_prob = 0.0
        count = 0   
        lm.compute_counts_kneser_key()
        print(f"Evaluating perplexity for {tokenizer} tokenizer with Kneser-Key smoothing")
        for line in tqdm(test_corpus):
            no_tokens = len(line)
            sentence = ["<s>"]*3 + line + ["</s>"]
            prob = lm.get_prob_kneser_key(sentence)
            total_prob += prob
            count += no_tokens+1

        corpus_perplexity = math.exp(-total_prob / max(count, 1))
        print(f"Average perplexity with Kneser-Key for {tokenizer}: {corpus_perplexity:.2f}")

        total_prob = 0.0
        count = 0
        print(f"Evaluating perplexity for {tokenizer} tokenizer with Witten-Bell smoothing")
        for line in tqdm(test_corpus):
            no_tokens = len(line)
            sentence = ["<s>"]*3 + line + ["</s>"]
            prob = lm.get_witten_bell_prob(sentence)
            total_prob += prob
            count += no_tokens+1

        corpus_perplexity = math.exp(-total_prob / max(count, 1))
        print(f"Average perplexity with Witten-Bell for {tokenizer}: {corpus_perplexity:.2f}")





