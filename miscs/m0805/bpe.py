from typing import List, Dict, Set
from itertools import chain

import re
from collections import defaultdict, Counter


def build_bpe(
        corpus: List[str],
        max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in descending manner.
    Hint: Counter in collection library would be helpful
    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are
          preprocessed together
    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab
    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token  # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token  # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token  # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token  # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token  # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END  # Use this token as the end of a word
    vocab_list = Counter(corpus)

    vocab = dict(
        [(str(' '.join(x[:-1])) + ' ' + str(x[-1] + ' _'), y) for (x, y) in
         vocab_list.items()])

    def get_pair_stat(vocab):
        stats = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                stats[symbols[i], symbols[i+1]] += freq
        return stats

    def replace_pair(most_freq, vocab):
        new_vocab = {}
        bigram = re.escape(' '.join(most_freq))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_token = pattern.sub(''.join(most_freq), word)
            new_vocab[new_token] = vocab[word]
        return new_vocab

    idx2word = []

    for token_list in vocab.keys():
        for token in token_list.split():
            if token not in idx2word:
                idx2word.append(token)

    for i in range(max_vocab_size):
        stats = get_pair_stat(vocab)
        if len(stats) == 0:
            break
        most_freq = max(stats, key=stats.get)
        print(most_freq)
        vocab = replace_pair(most_freq, vocab)
        idx2word.append(most_freq[0] + most_freq[1])
        if len(idx2word) == max_vocab_size-5:
            break

    idx2word = SPECIAL + sorted(idx2word, key=len, reverse=True)

    return idx2word


def encode(
        sentence: List[str],
        idx2word: List[str]
) -> List[int]:
    """ BPE encoder
    Implement byte pair encoder which takes a sentence and gives the encoded tokens
    Arguments:
    sentence -- The list of words which need to be encoded.
    idx2word -- The vocab that you have made on the above build_bpe function.

    Return:
    tokens -- The list of the encoded tokens
    """
    WORD_END = BytePairEncoding.WORD_END

    output_tokens = []
    for token in sentence:
        chars = list(token)
        sub_tokens = []
        end = len(chars)
        while end >= 1:
            start = 0
            cur_substr = None
            while start < end:
                substr = ''.join(chars[start:end])
                if end == len(chars):
                    substr = substr + '_'
                    if sub_tokens:
                        substr = substr[:-1]
                if substr in idx2word:
                    cur_substr = substr
                    break
                start += 1
                if start == end:
                    sub_tokens.append(idx2word.index('_'))
                    start = 0
            sub_tokens.append(idx2word.index(cur_substr))
            if cur_substr[-1] == '_':
                end -= len(cur_substr) - 1
            else:
                end -= len(cur_substr)
        output_tokens += sub_tokens[::-1]
    return output_tokens


def decode(
        tokens: List[int],
        idx2word: List[str]
) -> List[str]:
    """ BPE decoder
    Implement byte pair decoder which takes tokens and gives the decoded sentence.
    Arguments:
    tokens -- The list of tokens which need to be decoded
    idx2word -- the vocab that you have made on the above build_bpe function.
    Return:
    sentence  -- The list of the decoded words
    """
    WORD_END = BytePairEncoding.WORD_END

    sentence = []
    segment = ''
    for token in tokens:
        segment += idx2word[token]
        if idx2word[token][-1] == '_':
            sentence.append(segment[:-1])
            segment = ''
    return sentence


#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)


#############################################
# Testing functions below.                  #
#############################################


def test_build_bpe():
    print("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_', \
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w',
                    WORD_END} and \
           "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
        "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa',
                     'ab', 'a', 'b', WORD_END}, \
        "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
        "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")


def test_encoding():
    print("======Encoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    result = encode(['abbccc'], vocab)
    assert encode(['abbccc'], vocab) == [8, 9, 5, 10, 11], \
        "Your bpe encoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert len(encode(['aaaaaaaa', 'aaaaaaa'], vocab)) == 7, \
        "Your bpe encoding does not math expected result"
    print("The second test passed!")

    print("All 2 tests passed!")


def test_decoding():
    print("======Decoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    result = decode([8, 9, 5, 10, 11], vocab)
    assert decode([8, 9, 5, 10, 11], vocab) == ['abbccc'], \
        "Your bpe decoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert decode([5, 5, 8, 5, 6, 7, 8], vocab) == ['aaaaaaaa', 'aaaaaaa'], \
        "Your BPE decoding does not math expected result"
    print("The second test passed!")


def test_consistency():
    print("======Consistency Test Case======")
    corpus = ['this is test corpus .',
              'we will check the consistency of your byte pairing encoding .',
              'you have to pass this test to get full scores .',
              'we hope you to pass tests wihtout any problem .',
              'good luck .']

    vocab = build_bpe(
        chain.from_iterable(sentence.split() for sentence in corpus), 80)

    sentence = 'this is another sentence to test encoding and decoding .'.split()
    result = decode(encode(sentence, vocab), vocab)
    assert decode(encode(sentence, vocab), vocab) == sentence, \
        "Your BPE does not show consistency."
    print("The consistency test passed!")


if __name__ == "__main__":
    test_build_bpe()
    test_encoding()
    test_decoding()
    test_consistency()