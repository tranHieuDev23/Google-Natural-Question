import load_data
import analysis
import preprocess
import train
import sys

dataset_path = sys.argv[1]  # google-nq.gz
embedding_path = sys.argv[2]  # '../FastText/fasttext/cc.en.300.bin'
model_path = sys.argv[3]  # model.h5

embedding_dim = 300

embedding_model = preprocess.load_embedding(embedding_path, embedding_dim)

training_set, test_set = load_data.load_dataset(dataset_path, 50000, 5000)
print("Analysis training set:")
analysis.analysis_data(training_set)
print("Analysis test set:")
analysis.analysis_data(test_set)

question_length = 32
candidate_length = 128
model = train.load_model(
    model_path, question_length, candidate_length, embedding_dim)
model.summary()


def dataset_generator(dataset, batch_size):
    preprocess_params = (batch_size, question_length,
                         candidate_length, embedding_dim, embedding_model)
    return preprocess.x_y_generator(dataset, preprocess_params)


training_data_count = 100000
test_data_count = 10000
batch_size = 256
epochs = 10

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
training_step = (training_data_count + batch_size - 1) // batch_size
test_step = (test_data_count + batch_size - 1) // batch_size
model.fit_generator(generator=dataset_generator(training_set, batch_size),
                    steps_per_epoch=training_step,
                    validation_data=dataset_generator(test_set, batch_size),
                    validation_steps=test_step,
                    epochs=epochs)

model.save_weights(model_path)
