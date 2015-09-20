#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from itertools import chain
import random
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import accuracy_score

example_data_1 = {
        'g01': "BBDE", 'g02': "CBEDF", 'g03': "BBAA", 'g04': "ABCD",
        'g05': "ABBDF", 'g06': "ABC", 'g07': "ABBAA", 'g08': "ACBD",
        'g09': "DEBBA", 'g10': "AABC", 'g11': "AAAAF", 'g12': "CCDACD",
        'g13': "CCADB", 'g14': "CBAF", 'g15': "ABCD", 'g16': "CDBD"
}

example_data_2 = {
        'g01': "BBDE", 'g02': "CBEDF", 'g03': "BBAA", 'g04': "ABCD",
        'g05': "ABBDF", 'g06': "ABC", 'g07': "ABBAA", 'g08': "ACBD",
        'g09': "DEBBA", 'g10': "AABC", 'g11': "AAAAF", 'g12': "CCDACD",
        'g13': "CCADB", 'g14': "CBAF", 'g15': "ABCD", 'g16': "CDBD",
        'g17': "BBDE", 'g18': "CBEDF", 'g19': "BBAA", 'g20': "ABCD",
        'g21': "ABBDF", 'g22': "ABC", 'g23': "ABBAA", 'g24': "ACBD",
        'g25': "DEBBA", 'g26': "AABC", 'g27': "AAAAF", 'g28': "CCDACD",
        'g29': "CCADB", 'g30': "CBAF", 'g31': "ABCD", 'g32': "CDBD"
}


def absolute_class_counts(data, expected_classes=None):
    """input: a dict mapping a group key to a list of class occurrences
              [0,2,2,1,0,1,2,2,0]
       output: a dict mapping class keys to their absolute counts
              {0:3, 1:2, 2:4}"""
    counts_class = defaultdict(int)
    if expected_classes is not None:
        for c in expected_classes:
            counts_class[c]
    for e in data:
        counts_class[e] += 1
    return counts_class


def relative_class_counts(data):
    """input: a dict mapping class keys to their absolute counts
       output: a dict mapping class keys to their relative counts"""
    counts_items = sum(data.values())
    return {k: 1.0 * v / counts_items for k, v in data.iteritems()}


def diff_distribution(a, b, weights=None):
    """compares two distributions and returns a sum of all (weighted)
    diffs"""
    assert a.keys() == b.keys()
    if weights is not None:
        assert a.keys() == weights.keys()
        diff = {k: weights[k] * abs(a[k] - b[k]) for k in a}
    else:
        diff = {k: abs(a[k] - b[k]) for k in a}
    return sum(diff.values())


def join_distributions(a, b):
    """joins two distributions of absolute class counts by adding the values
    of each key"""
    assert a.keys() == b.keys()
    return {k: a[k] + b[k] for k in a}


class GroupwiseStratifiedKFold(object):
    # TODO: save and report final diffs from class distribution

    def __init__(self, number_of_folds, data, shuffle=False, seed=0):
        self.fold_register = {}
        ungrouped_data = list(chain(*data.values()))
        counts_class_absolute = absolute_class_counts(ungrouped_data)
        counts_class_relative = relative_class_counts(counts_class_absolute)
        classes = list(counts_class_absolute.keys())
        class_weights = {k: 1-v for k, v in counts_class_relative.iteritems()}
        group_distribution = {k: absolute_class_counts(
                                    list(v), expected_classes=classes)
                              for k, v in data.iteritems()}
        folds = {n: {k: 0 for k in counts_class_relative}
                 for n in range(1, number_of_folds + 1)}
        fold_register = {n: [] for n in folds.keys()}
        pool = set(group_distribution.keys())

        cnt_pass = 0
        while len(pool) > 0:
            # either shuffle the order of filling folds in this pass randomly
            # or rotate it, in order to prevent that the first folds or a pass
            # always get the best possible draw from the pool
            if shuffle:
                random.seed(seed + cnt_pass)
                fold_order_in_this_pass = folds.keys()
                random.shuffle(fold_order_in_this_pass)
            else:
                fold_order_in_this_pass = deque(folds.keys())
                fold_order_in_this_pass.rotate(-cnt_pass)

            # in a pass, fill each fold with the best group
            for this_fold in fold_order_in_this_pass:
                this_folds_dist = folds[this_fold]
                if len(pool) == 0:
                    break

                # find the group in the pool, that minimizes the difference of
                # this fold to the base distribution
                min_diff = float('+inf')
                min_group = None
                min_joint_dist = None
                for group in pool:
                    joint_dist = join_distributions(this_folds_dist,
                                                    group_distribution[group])
                    diff = diff_distribution(counts_class_relative,
                                             relative_class_counts(joint_dist),
                                             weights=class_weights)
                    if diff < min_diff:
                        min_diff = diff
                        min_group = group
                        min_joint_dist = joint_dist

                # remove group from pool, register group in fold and add group
                # absolutes to fold
                pool.remove(min_group)
                fold_register[this_fold].append(min_group)
                folds[this_fold] = min_joint_dist

            cnt_pass += 1

        self.fold_register = fold_register

    def __iter__(self):
        for test_fold in self.fold_register.keys():
            train_foldes = list(self.fold_register.keys())
            train_foldes.remove(test_fold)
            l = [self.fold_register[f] for f in train_foldes]
            train_ids = list(chain(*l))
            test_ids = self.fold_register[test_fold]
            yield train_ids, test_ids


class RepeatedGroupwiseStratifiedKFold():

    def __init__(self, number_of_folds, data, shuffle=False, seed=0,
                 repeats=10):
        self.iterations = []
        for repeat_nr in range(repeats):
            foldes = GroupwiseStratifiedKFold(
                number_of_folds, data, shuffle=shuffle, seed=seed + repeat_nr)
            for fold_nr, (train, test) in enumerate(foldes):
                self.iterations.append((train, test,
                                        '%d-%d' % (repeat_nr, fold_nr)))

    def __iter__(self):
        for train_ids, test_ids, iteration_id in self.iterations:
            yield train_ids, test_ids, iteration_id


def load_corpus(path, silent=True):
    from arggraph import ArgGraph
    from os import walk
    r = {}
    _, _, filenames = walk(path).next()
    for i in filenames:
        if i.endswith('.xml'):
            if not silent:
                print i, '...'
            g = ArgGraph()
            g.load_from_xml(path + i)
            r[g.graph['id']] = g
    return r


def build_kfold_reference_dataset(graph_path):
    # the more sophisticated CV splitter needs a good labeling to produce
    # similar folds. I want to use the role+complextype labelling here,
    # which is not in the datafiles loaded above. I thus extract it from
    # the corpus of files directly.

    corpus = load_corpus(graph_path)
    data = {tid: graph.get_role_type_labels().values()
            for tid, graph in corpus.iteritems()}
    return data


class Densifier(BaseEstimator):
    ''' Lars Buitinck
    http://sourceforge.net/p/scikit-learn/mailman/message/30427864/
    '''

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return X.toarray()


def evaluate(ground_truth, prediction, labels=None):
    def prfs_to_dict(l):
        return {'precision': l[0], 'recall': l[1], 'fscore': l[2]}

    results = {}
    items_count = len(ground_truth)

    # accuracy
    accuracy = accuracy_score(ground_truth, prediction)
    results['accuracy'] = accuracy

    # confusion matrix
    conf_matr = confusion_matrix(ground_truth, prediction)
    categories = set(ground_truth) | set(prediction)
    confusions = {gold: {pred: conf_matr[gold][pred] for pred in categories}
                  for gold in categories}
    results['confusions'] = confusions

    # class wise PRF
    classwise = precision_recall_fscore_support(ground_truth, prediction,
                                                average=None)
    results['true_cat_dist'] = classwise[-1]
    results['classwise'] = {cl: prfs_to_dict([classwise[0][cl],
                                              classwise[1][cl],
                                              classwise[2][cl]])
                            for cl in categories}  # TODO: labels

    # average PRF
    results['macro_avg'] = prfs_to_dict(
        precision_recall_fscore_support(ground_truth, prediction,
                                        average='macro', pos_label=None))
    results['micro_avg'] = prfs_to_dict(
        precision_recall_fscore_support(ground_truth, prediction,
                                        average='micro', pos_label=None))
    results['weigh_avg'] = prfs_to_dict(
        precision_recall_fscore_support(ground_truth, prediction,
                                        average='weighted', pos_label=None))

    # marginals
    gold_category_distribution = {
        g: sum([confusions[g][p] for p in categories]) for g in categories
    }
    pred_category_distribution = {
        p: sum([confusions[g][p] for g in categories]) for p in categories
    }

    # kappa
    expected_agreement_fleiss = sum([
        ((gold_category_distribution[c] + pred_category_distribution[c]) /
         (2.0 * items_count)) ** 2
        for c in categories
    ])
    expected_agreement_cohen = sum([
        (float(gold_category_distribution[c]) / items_count) *
        (float(pred_category_distribution[c]) / items_count)
        for c in categories
    ])
    kappa_fleiss = (1.0 * (accuracy - expected_agreement_fleiss) /
                    (1 - expected_agreement_fleiss))
    kappa_cohen = (1.0 * (accuracy - expected_agreement_cohen) /
                   (1 - expected_agreement_cohen))
    results['k_fleiss'] = {'k': kappa_fleiss,
                           'AE': expected_agreement_fleiss,
                           'AO': accuracy}
    results['k_cohen'] = {'k': kappa_cohen,
                          'AE': expected_agreement_cohen,
                          'AO': accuracy}

    return results


if __name__ == '__main__':
    for train, test in GroupwiseStratifiedKFold(5, example_data_2,
                                                shuffle=True, seed=1):
        print test

    l = list(GroupwiseStratifiedKFold(5, example_data_2, shuffle=True, seed=1))
    print l
