
import numpy as np

from minimal_is_all_you_need.models import Bert, GPT, GPT_2, Transformer, GPT_3
from minimal_is_all_you_need.elmo import ELMo
from minimal_is_all_you_need.xlnet import XLNet
from minimal_is_all_you_need.bert import masked_perplexity, the_loss_of_bert
from minimal_is_all_you_need.transformer_xl import TransformerXL


def get_example_data():
    X1 = np.random.random((100, 100))
    X2 = np.random.random((100, 100))
    Y1 = np.random.random((100, 100, 100))
    Y2 = np.random.random((100, 1))
    return [X1, X2], [Y1, Y2]