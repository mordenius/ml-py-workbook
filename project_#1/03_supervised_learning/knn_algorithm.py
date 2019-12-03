import datasets.dataset_provider as data_provider
import random
import math
import operator


def load_dataset(split, training_set=[], test_set=[]):
    dataset = data_provider.get_iris()
    for index in range(dataset.shape[0]):
        if random.random() < split:
            training_set.append(dataset.loc[index])
        else:
            test_set.append(dataset.loc[index])


def get_euclidean_distance(source, destination, number_of_axles=2):
    distance = 0
    for axis in range(number_of_axles):
        distance += pow((source[axis] - destination[axis]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, number_of_neighbors):
    distance = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = get_euclidean_distance(test_instance, training_set[x], length)
        distance.append((training_set[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(number_of_neighbors):
        neighbors.append(distance[x][0])
    return neighbors


def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] is predictions[x]:
            correct += 1

    return (correct / float(len(test_set))) * 100.0


if __name__ == '__main__':
    training_set = []
    test_set = []
    load_dataset(0.66, training_set, test_set)
    print("Train: {}".format(len(training_set)))
    print("Test: {}".format(len(test_set)))

    predictions = []
    number_of_neighbors = 3

    for x in range(len(test_set)):
        neighbors = get_neighbors(
            training_set, test_set[x], number_of_neighbors)
        response = get_response(neighbors)
        predictions.append(response)
        print('> predicted: %s, actual is %s' %
              (repr(response), repr(test_set[x][-1])))

    accuracy = get_accuracy(test_set, predictions)
    print("Accuracy: {}".format(accuracy))
