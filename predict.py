import preprocess
import train
import numpy as np

question_length = 32
candidate_length = 128
embedding_dim = 128
embedding_path = '../FastText/fasttext/cc.en.300.bin'

embedding_model = preprocess.load_embedding(embedding_path, embedding_dim)
model = train.load_model('model.h5', question_length, candidate_length, embedding_dim)

while True:
    question = input("Input your question: ")
    question = question.split(' ')
    question = preprocess.remove_html_tags(question)
    question = preprocess.get_embedding(question, question_length, embedding_model)

    candidate = input("Input your candidate: ")
    candidate = candidate.split(' ')
    candidate = preprocess.remove_html_tags(candidate)
    candidate = preprocess.get_embedding(candidate, candidate_length, embedding_model)

    X_q = np.asarray([question])
    X_c = np.asarray([candidate])
    y = model.predict([X_q, X_c])
    print("Prediction: " + str(y))
