import pandas as pd
import sacrebleu
import sentencepiece as spm


def load_vocab_model(file_path):
    tokenizer = spm.SentencePieceProcessor(model_file=file_path)
    return tokenizer

def decode_tokens(file_path, tokenizer):
    with open(file_path) as f:
        # remove dot and space from the end of the sentence
        return [tokenizer.decode_pieces(line.strip().split()[:-1]) for line in f]
    
    
test_file_path = 'PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
test_df = pd.read_csv(test_file_path, sep='|')
reference_translations = [sign_trans.strip() for sign_trans in test_df['translation'].tolist()]
print(f"Length of reference translations: {len(reference_translations)}")
# show five last reference translations
print(f"Reference translations: {reference_translations[-5:]}")
tokenizer_path = 'vocabs/phoenix2014t-2000.model'
tokenizer = load_vocab_model(tokenizer_path)

hypothesis_translation_path = 'pred.txt'
hypothesis_translations = decode_tokens(hypothesis_translation_path, tokenizer)
print(f"length of hypothesis translations: {len(hypothesis_translations)}")
print(f"Hypothesis translations: {hypothesis_translations[-5:]}")

# Compute BLEU score
bleu_score = sacrebleu.corpus_bleu(hypothesis_translations, [reference_translations])

print("BLEU Score:", bleu_score)