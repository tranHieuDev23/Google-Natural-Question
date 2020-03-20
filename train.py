from os import path
import pickle
from keras.layers import Input, Bidirectional, GRU, Dense, Dropout, concatenate
from keras.models import Model


def build_model(question_length, candidate_length, embedding_length):
    inputQ = Input(shape=(question_length, embedding_length))
    inputC = Input(shape=(candidate_length, embedding_length))

    q0 = Bidirectional(GRU(256))(inputQ)
    q1 = Dropout(0.2)(q0)
    q2 = Dense(256, activation='relu')(q1)
    q3 = Dropout(0.2)(q2)

    c0 = Bidirectional(GRU(256))(inputC)
    c1 = Dropout(0.2)(c0)
    c2 = Dense(256, activation='relu')(c1)
    c3 = Dropout(0.2)(c2)

    a0 = concatenate([q1, c1])
    z1 = Dense(256, activation='relu')(a0)
    a1 = Dropout(0.2)(z1)
    z2 = Dense(256, activation='relu')(a1)
    a2 = Dropout(0.2)(z2)
    z3 = Dense(256, activation='relu')(a2)
    a3 = Dropout(0.2)(z3)
    output = Dense(1, activation='sigmoid')(a3)

    return Model(inputs=[inputQ, inputC], outputs=output)

def load_model(model_path, question_length, candidate_length, embedding_length):
    model = build_model(question_length, candidate_length, embedding_length)
    try:
        model.load_weights(model_path)
    except Exception as e:
        pass
    return model
