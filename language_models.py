from utils import load_tokenized_jsonl
from tqdm import tqdm 
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

if __name__ == "__main__":
    # Example usage
    corpus = load_tokenized_jsonl("cache/eng_train_bpe.jsonl")
    print(f"Loaded corpus with {len(corpus)} lines.")
    lm = LM()
    lm.train(corpus)
    lm.predict("Once upon a time")




