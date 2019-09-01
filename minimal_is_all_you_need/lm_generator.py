
import keras
import numpy as np

class LMDataGenerator(keras.utils.Sequence):

    def __len__(self):
        return int(np.ceil(len(self.indices)/self.batch_size))

    def __init__(self, corpus, vocab, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True, token_encoding='word'):

        self.corpus = corpus
        self.vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab).readlines()}
        self.sent_ids = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sentence_maxlen = sentence_maxlen
        self.token_maxlen = token_maxlen
        self.token_encoding = token_encoding
        with open(self.corpus) as fp:
            self.indices = np.arange(len(fp.readlines()))
            newlines = [index for index in range(0, len(self.indices), 2)]
            self.indices = np.delete(self.indices, newlines)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        if self.token_encoding == 'char': word_char_indices_batch = np.full((len(batch_indices), self.sentence_maxlen, self.token_maxlen), 260, dtype=np.int32)
        for i, batch_id in enumerate(batch_indices):
            word_indices_batch[i] = self.get_token_indices(sent_id=batch_id)
            if self.token_encoding == 'char': word_char_indices_batch[i] = self.get_token_char_indices(sent_id=batch_id)
        for_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        padding = np.zeros((1,), dtype=np.int32)
        for i, word_seq in enumerate(word_indices_batch ): for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)
        for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]
        back_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        for i, word_seq in enumerate(word_indices_batch): back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)
        back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]
        return [word_indices_batch if self.token_encoding == 'word' else word_char_indices_batch, for_word_indices_batch, back_word_indices_batch], []

    def on_epoch_end(self):
        if self.shuff: np.random.shuffle(self.indices)

    def get_token_indices(self, sent_id):
        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen,), dtype=np.int32)
                    token_ids[0] = self.vocab['<bos>']
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
                        if token.lower() in self.vocab:
                            token_ids[j + 1] = self.vocab[token.lower()]
                        else:
                            token_ids[j + 1] = self.vocab['<unk>']
                    if token_ids[1]:
                        token_ids[j + 2] = self.vocab['<eos>']
                    return token_ids

    def get_token_char_indices(self, sent_id):
        def convert_token_to_char_ids(token, token_maxlen):
            bos_char = 256  # <begin sentence>
            eos_char = 257  # <end sentence>
            bow_char = 258  # <begin word>
            eow_char = 259  # <end word>
            pad_char = 260  # <pad char>
            char_indices = np.full([token_maxlen], pad_char, dtype=np.int32)
            word_encoded = token.encode('utf-8', 'ignore')[:(token_maxlen - 2)]
            char_indices[0] = bow_char
            if token == '<bos>':
                char_indices[1] = bos_char
                k = 1
            elif token == '<eos>':
                char_indices[1] = eos_char
                k = 1
            else:
                for k, chr_id in enumerate(word_encoded, start=1): char_indices[k] = chr_id + 1
            char_indices[k + 1] = eow_char
            return char_indices

        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen, self.token_maxlen), dtype=np.int32)
                    token_ids[0] = convert_token_to_char_ids('<bos>', self.token_maxlen)
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]): token_ids[j + 1] = convert_token_to_char_ids(token, self.token_maxlen)
                    if token_ids[1]: token_ids[j + 2] = convert_token_to_char_ids('<eos>', self.token_maxlen)

        return token_ids
