
import numpy as np

from minimal_is_all_you_need import Transformer, ELMo, Bert, GPT, GPT_2, XLNet, TransformerXL, the_loss_of_bert, get_example_data

X, Y = get_example_data()


def main():


    model = Bert()
    model.compile('adam', loss=[the_loss_of_bert(0.1), 'binary_crossentropy'])
    model.fit(X, Y)
    model.predict(X)


    # X1 = np.random.random((2, 3))
    # X2 = np.random.random((2, 1))
    # Y2 = np.random.random((2, 3, 1))
    # model = TransformerXL()
    # model.compile('adam', loss='sparse_categorical_crossentropy')
    # model.fit([X1,X2], Y2, batch_size=2)


    # X1 = np.random.random((100, 100))
    # X2 = np.random.random((100, 100))
    # Y1 = np.random.random((100, 100, 1))
    # Y2 = np.random.random((100, 1))
    # model = Transformer()
    # model.compile('adam', loss='sparse_categorical_crossentropy')
    # model.fit(X1, Y1)
    # model.predict(X2)


    # X1 = np.random.random((100, 100))
    # X2 = np.random.random((100, 100))
    # Y1 = np.random.random((100, 100, 100))
    # Y2 = np.random.random((100, 1))
    # model = GPT()
    # model.compile('adam', loss='sparse_categorical_crossentropy')
    # model.fit(X1, Y1)
    # model.predict(X2)


    # X1 = np.random.random((100, 100))
    # X2 = np.random.random((100, 100))
    # Y1 = np.random.random((100, 100, 1))
    # Y2 = np.random.random((100, 1))
    # model = GPT_2()
    # model.compile('adam', loss='sparse_categorical_crossentropy')
    # model.fit(X1, Y1)
    # model.predict(X2)


    # X1 = np.random.random((100, 100))
    # X2 = np.random.random((100, 100, 1))
    # Y1 = np.random.random((100, 100, 1))
    # Y2 = np.random.random((100, 100))
    # model = ELMo()
    # model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy')
    # model.fit([X1, Y1, X2])
    # model.predict(X)


    # i = 32 
    # X = [np.random.random((i, 100)), np.random.random((i, 100)), np.random.random((i, 1)), np.random.random((i, 100))]
    # X = [np.random.random((i, 100)), np.random.random((i, 100)), np.random.random((i, 1))] #if training=False
    # Y1 = np.random.random((i, 100, 1))
    # model = XLNet(target_len=X[0].shape[1])
    # model.summary()
    # model.compile('adam', loss='sparse_categorical_crossentropy')
    # model.fit(X, Y1)
    # model.predict(X)


main()
 