import pandas
import load_data


def analysis_data(dataset):
    (document_texts, question_texts, candidates, is_answers) = dataset
    allCandidate = []
    for l in candidates:
        allCandidate.extend(l)
    documentFrame = pandas.DataFrame(
        map(lambda item: len(item), document_texts))
    questionFrame = pandas.DataFrame(
        map(lambda item: len(item), question_texts))
    candidateFrame = pandas.DataFrame(
        map(lambda item: item[1] - item[0], allCandidate))
    answerFrame = pandas.DataFrame(is_answers)
    print(documentFrame.describe())
    print(questionFrame.describe())
    print(candidateFrame.describe())
    print(answerFrame.describe())
