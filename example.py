

from minimal_is_all_you_need import Transformer, ELMo, Bert, GPT, GPT_2, XLNet, the_loss_of_bert, get_example_data

X, Y = get_example_data()


def main():

    model = Bert()

    model.compile('adam', loss=[the_loss_of_bert(0.1), 'binary_crossentropy'])
    model.fit(X, Y)
    model.predict(X)

    model = Transformer()
    model.compile('adam', loss='sparse_categorical_crossentropy')

    model = GPT()
    model.compile('adam', loss='sparse_categorical_crossentropy')

    model = GPT_2()
    model.compile('adam', loss='sparse_categorical_crossentropy')

    model = ELMo()
    model.compile('adagrad', loss=None)

    model = XLNet()
    model.compile('adam', loss='sparse_categorical_crossentropy')

main()
