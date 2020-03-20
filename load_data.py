import gzip
import jsonlines
from preprocess import remove_html_tags


def __extract_candidates__(line):
    all_candidates = []
    for item in line['long_answer_candidates']:
        start = item['start_token']
        end = item['end_token']
        all_candidates.append((start, end))
    n = len(all_candidates)
    is_answers = [0] * n
    limit = 0
    for item in line['annotations']:
        if ('long_answer' not in item):
            continue
        long_answer = item['long_answer']
        candidate_index = long_answer['candidate_index']
        if (0 <= candidate_index and candidate_index < n):
            is_answers[candidate_index] = 1
            limit += 1
    limit = min(limit, n - limit)
    cnt = [0, 0]
    selected_candidates = []
    selected_answers = []
    for i in range(n):
        if (cnt[is_answers[i]] >= limit):
            continue
        cnt[is_answers[i]] += 1
        selected_candidates.append(all_candidates[i])
        selected_answers.append(is_answers[i])
    return selected_candidates, selected_answers


def load_dataset(file_path, training_count, test_count):
    training = None
    test = None
    with gzip.open(file_path) as gf:
        reader = jsonlines.Reader(gf)
        document_texts = []
        question_texts = []
        candidates = []
        is_answers = []
        line_count = 0
        for line in reader:
            candidates_i, is_answer_i = __extract_candidates__(line)
            if (len(is_answer_i) == 0):
                continue
            document_texts.append(line['document_text'].split(' '))
            question_texts.append(line['question_text'].split(' '))
            candidates.append(candidates_i)
            is_answers.append(is_answer_i)
            line_count += 1
            if (line_count == training_count):
                training = (document_texts, question_texts,
                            candidates, is_answers)
                document_texts = []
                question_texts = []
                candidates = []
                is_answers = []
            elif (line_count == training_count + test_count):
                test = (document_texts, question_texts, candidates, is_answers)
                break
    return training, test
