import fasttext
import fasttext.util
from random import shuffle
import numpy as np

html_tags = set(['<p>', '<table>', '<tr>', '<ul>',
                 '<ol>', '<dl>', '<li>', '<dd>', '<dt>'])


def load_embedding(embedding_path, embedding_dim):
    model = fasttext.load_model(embedding_path)
    if (model.get_dimension() > embedding_dim):
        fasttext.util.reduce_model(model, embedding_dim)
    return model


def remove_html_tags(tokens):
    return [token for token in tokens if token not in html_tags]


def get_embedding(tokens, output_length, model):
    l = len(tokens)
    if (l < output_length):
        tokens.extend([''] * (output_length - l))
    elif (l > output_length):
        tokens = tokens[:output_length]
    return [model.get_word_vector(token) for token in tokens]


def x_y_generator(dataset, preprocess_params):
    (document_texts, question_texts, candidates, is_answers) = dataset
    (batch_size, question_length, candidate_length, embedding_dim, model) = preprocess_params
    X_q = []
    X_c = []
    y = []
    item_count = 0
    input_size = len(is_answers)
    i = -1
    while(True):
        i = (i + 1) % input_size
        document_text = document_texts[i]
        question_text = question_texts[i]
        candidates_i = candidates[i]
        is_answers_i = is_answers[i]
        question_vector = get_embedding(
            question_text, question_length, model)
        for j in range(len(candidates_i)):
            start, end = candidates_i[j]
            non_html_tokens = remove_html_tags(
                document_text[start:end])
            if (len(non_html_tokens) > candidate_length):
                non_html_tokens = non_html_tokens[:candidate_length]
            candidate_vector = get_embedding(
                non_html_tokens, candidate_length, model)
            X_q.append(question_vector)
            X_c.append(candidate_vector)
            y.append(is_answers_i[j])
            item_count += 1
            if (item_count == batch_size):
                yield [np.asarray(X_q), np.asarray(X_c)], np.asarray(y)
                X_q = []
                X_c = []
                y = []
                item_count = 0
    if (item_count > 0):
        yield [np.asarray(X_q), np.asarray(X_c)], np.asarray(y)
