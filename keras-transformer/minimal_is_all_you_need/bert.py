

import random
import numpy as np

from keras import backend as K
from typing import List, Callable
from itertools import islice, chain
from keras.utils import get_custom_objects


class BatchGeneratorForBERT:

    reserved_positions = 3
    def __init__(self, sampler, dataset_size, sep_token_id, cls_token_id, mask_token_id, first_normal_token_id, last_normal_token_id, sequence_length, batch_size, sentence_min_span=0.25):

        self.sampler = sampler
        self.steps_per_epoch = ( dataset_size // (sequence_length * batch_size))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.mask_token_id = mask_token_id
        self.first_token_id = first_normal_token_id
        self.last_token_id = last_normal_token_id
        assert 0.0 < sentence_min_span <= 1.0
        self.sentence_min_length = max( int(sentence_min_span * (self.sequence_length - self.reserved_positions)), 1)
        self.sentence_max_length = ( self.sequence_length - self.reserved_positions - self.sentence_min_length)

    def generate_batches(self):
        samples = self.generate_samples()
        while True:
            next_bunch_of_samples = islice(samples, self.batch_size)
            has_next, mask, sequence, segment, masked_sequence = zip( *list(next_bunch_of_samples))
            combined_label = np.stack([sequence, mask], axis=-1)
            yield ( [np.array(masked_sequence), np.array(segment)], [combined_label, np.expand_dims(np.array(has_next, dtype=np.float32), axis=-1)] )

    def generate_samples(self):

        while True:
            a_length = random.randint(
                self.sentence_min_length,
                self.sentence_max_length)
            b_length = (
               self.sequence_length - self.reserved_positions - a_length)

            has_next = random.random() < 0.5
            if has_next:
                full_sample = self.sampler(a_length + b_length)
                sentence_a = full_sample[:a_length]
                sentence_b = full_sample[a_length:]
            else:
                sentence_a = self.sampler(a_length)
                sentence_b = self.sampler(b_length)
            assert len(sentence_a) == a_length
            assert len(sentence_b) == b_length
            sequence = (
                [self.cls_token_id] +
                sentence_a + [self.sep_token_id] +
                sentence_b + [self.sep_token_id])
            masked_sequence = sequence.copy()
            output_mask = np.zeros((len(sequence),), dtype=int)
            segment_id = np.full((len(sequence),), 1, dtype=int)
            segment_id[:a_length + 2] = 0
            for word_pos in chain(
                    range(1, a_length + 1),
                    range(a_length + 2, a_length + 2 + b_length)):
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_sequence[word_pos] = self.mask_token_id
                    elif dice < 0.9:
                        masked_sequence[word_pos] = random.randint(
                            self.first_token_id, self.last_token_id)
                    output_mask[word_pos] = 1
            yield (int(has_next), output_mask, sequence,
                   segment_id, masked_sequence)


def masked_perplexity(y_true, y_pred):

    y_true_value = y_true[:, :, 0]
    mask = y_true[:, :, 1]
    cross_entropy = K.sparse_categorical_crossentropy(y_true_value, y_pred)
    batch_perplexities = K.exp(
        K.sum(mask * cross_entropy, axis=-1) / (K.sum(mask, axis=-1) + 1e-6))
    return K.mean(batch_perplexities)


class the_loss_of_bert:

    def __init__(self, penalty_weight=0.1):
        self.penalty_weight = penalty_weight

    def __call__(self, y_true, y_pred):
        y_true_val = y_true[:, :, 0]
        mask = y_true[:, :, 1]

        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        masked_cross_entropy = ( K.sum(mask * K.sparse_categorical_crossentropy(y_true_val, y_pred), axis=-1) / num_items_masked)
        masked_entropy = ( K.sum(mask * -K.sum(y_pred * K.log(y_pred), axis=-1), axis=-1) / num_items_masked)
        return masked_cross_entropy - self.penalty_weight * masked_entropy

    def get_config(self):
        return { 'penalty_weight': self.penalty_weight }


get_custom_objects().update({
    'MaskedPenalizedSparseCategoricalCrossentropy':
        the_loss_of_bert,
    'masked_perplexity': masked_perplexity,
})
