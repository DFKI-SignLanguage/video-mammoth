import codecs
import collections
import itertools
import os

from mammoth.utils.logging import logger
from mammoth.constants import DefaultTokens
import sentencepiece as spm


DEFAULT_SPECIALS = (DefaultTokens.BOS, DefaultTokens.EOS, DefaultTokens.UNK, DefaultTokens.PAD)


def get_vocab(path):
    new_vocab = Vocab(path, items=None)
    logger.debug(new_vocab)
    return new_vocab


class Vocab():
    def __init__(self, model_path, items=None):
        # Load SentencePiece model
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        self.path = model_path
        self.itos = []
        self.stoi = collections.defaultdict(int)
        self.specials = list(DEFAULT_SPECIALS)
        if items is None:
            items = [self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.get_piece_size())]
        
        # remove specials from items
        items = [i for i in items if i not in self.specials]
        self.add_token(items + self.specials)
        assert len(self.stoi) == len(self.itos)
        
    def __getitem__(self, key_str):
        return self.stoi[key_str]

    def __len__(self):
        return len(self.stoi)


    # TODO: likely can be deleted
    def add_token(self, tokens):
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    @classmethod
    def merge(cls, *vocabs, size=None):
        """Merge vocabs."""
        specials = collections.OrderedDict()
        for vocab in vocabs:
            for elem in vocab.specials:
                specials[elem] = None
        specials = [elem for elem in specials.keys()]

        # from itertools recipes https://docs.python.org/3/library/itertools.html#itertools-recipes
        # FIXME: would rather install more-itertools, but for now I'm trying to keep deps as minimal as possible
        def roundrobin(*iterables):
            "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
            # Recipe credited to George Sakkis
            num_active = len(iterables)
            nexts = itertools.cycle(iter(it).__next__ for it in iterables)
            while num_active:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    # Remove the iterator we just exhausted from the cycle.
                    num_active -= 1
                    nexts = itertools.cycle(itertools.islice(nexts, num_active))

        items = list(roundrobin(*[vocab.stoi.keys() for vocab in vocabs]))
        return cls(None, items=items, tag='', size=size, specials=specials)

    def __repr__(self):
        return f"{self.__class__.__name__} @ {self.path} ({len(self)} items, specials=[{self.specials}])"


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.
    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line, may followed with a count number
            seperate by space if `with_count`. No extra whitespace is allowed.
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            first_line = lines[0].split(None, 1)
            has_count = (len(first_line) == 2 and first_line[-1].isdigit())
            if has_count:
                vocab = [line.split(None, 1) for line in lines]
                orig_len = len(vocab)
                vocab = [tpl for tpl in vocab if len(tpl) == 2]
                if len(vocab) != orig_len:
                    logger.warning(f'Dropped invalid entries from {vocab_path}')
            else:
                vocab = [line.strip().split()[0] for line in lines]
            return vocab, has_count
