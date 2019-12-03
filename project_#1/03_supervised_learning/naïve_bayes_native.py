import datasets.dataset_provider as data_provider
import math
import random

from sklearn.model_selection import train_test_split


def load_dataset():
    dataset = data_provider.get_pima()
    return dataset.values.tolist()


def split_dataset(dataset, split_ratio):
    next_dataset_size = int(len(dataset) * split_ratio)
    next_dataset = []
    copy = list(dataset)
    while len(next_dataset) < next_dataset_size:
        index = random.randrange(len(copy))
        next_dataset.append(copy.pop(index))
    return (next_dataset, copy)


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        vector_class = vector[-1]
        if vector_class not in separated:
            separated[vector_class] = []
        separated[vector_class].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
    return (1/(math.sqrt(2*math.pi) * stdev)) * exponent


def calculate_class_probability(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
        return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probability(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1

    return (correct / float(len(test_set))) * 100.0


if __name__ == '__main__':
    dataset = load_dataset()

    train_set, test_set = split_dataset(dataset, 0.67)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(
        len(dataset), len(train_set), len(test_set)))

    summaries = summarize_by_class(train_set)
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))
