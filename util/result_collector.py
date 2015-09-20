#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import strftime
import numpy as np
import pandas as pd
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
from itertools import product
from random import shuffle
from sklearn.base import BaseEstimator


def is_numeric(obj):
    # http://stackoverflow.com/a/500908
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def value_in_nested_dict(d, path_of_keys):
    ''' follows a path of keys in a nested dict and returns the value found
        at the end of the path '''
    if len(path_of_keys) == 0:
        return d
    else:
        try:
            return value_in_nested_dict(d[path_of_keys[0]], path_of_keys[1:])
        except KeyError:
            return None


def filter_params(params):
    ''' this function can be used to filter the get_params output for
        Estimator instances, so that no objects but only their string
        representations are pickled '''
    out = {}
    for k, v in params.iteritems():
        if isinstance(v, dict):
            v2 = filter_params(v)
        elif isinstance(v, BaseEstimator) or hasattr(v, '__call__'):
            v2 = str(v).replace('\n      ', '')
        else:
            v2 = v
        out[k] = v2
    return out


def load_result_collector(filename):
    '''loads a dumped collected results from a gzip pickle file'''
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


class result_collector():

    def __init__(self, name='noname', series='noseries', desc='nodesc'):
        # meta data
        self.name = name
        self.series = series
        self.desc = desc
        self.timestamp = strftime("%Y-%m-%d_%X")
        # registers
        self.conditions = list()
        self.levels = list()
        self.iterations = list()
        # data store
        self.data = list()
        self.path_to_metric = list()

    def add_result(self, condition, iteration, level, data):
        if condition not in self.conditions:
            self.conditions.append(condition)
        if iteration not in self.iterations:
            self.iterations.append(iteration)
        if level not in self.levels:
            self.levels.append(level)
        # TODO: check for double entries
        self.data.append((condition, iteration, level, data))

    def _get_result(self, condition, level):
        assert condition in self.conditions
        assert level in self.levels
        relevant_data = [value_in_nested_dict(d, self.path_to_metric)
                         for c, _i, l, d in self.data
                         if c == condition and l == level]
        return relevant_data

    def _sum_result(self, condition, level):
        relevant_data = self._get_result(condition, level)
        return pd.Series(relevant_data).describe()

    def print_result(self, condition, level):
        print self._sum_result(condition, level)

    def _string_summary(self, condition, level):
        t = self._sum_result(condition, level)
        return "%.3f (+- %.3f)" % (t['mean'], t['std'])

    def print_all_results(self):
        # print header
        print '\t'.join(self.conditions)
        for level in self.levels:
            print '\t'.join([level] + [self._string_summary(condition, level)
                                       for condition in self.conditions])

    def print_all_results_for_level(self, level):
        print '\t'.join(self.conditions)
        print '\t'.join([level] + [self._string_summary(condition, level)
                                   for condition in self.conditions])

    def set_metric(self, path_of_keys, ignore_type=False):
        # test path first
        if len(self.data) > 0:
            v = value_in_nested_dict(self.data[0][3], path_of_keys)
            if (not ignore_type) and (not is_numeric(v)):
                raise ValueError(('The path of keys does not lead to a'
                                  'numerical object/number.'))
        else:
            print "Warning: path_of_keys to the metric cannot be validated."
        self.path_to_metric = path_of_keys

    def plot_compare_conditions(self, condition_a, condition_b, level):
        '''generates two boxplots'''
        assert condition_a in self.conditions
        assert condition_b in self.conditions
        data_a = [value_in_nested_dict(d, self.path_to_metric)
                  for c, _i, l, d in self.data
                  if c == condition_a and l == level]
        data_b = [value_in_nested_dict(d, self.path_to_metric)
                  for c, _i, l, d in self.data
                  if c == condition_b and l == level]
        plt.boxplot([data_a, data_b])
        plt.scatter([1, 2], [np.mean(data_a), np.mean(data_b)])
        plt.xticks([1, 2], [condition_a, condition_b])
        plt.show()

    def plot_compare_all_conditions(self, level):
        '''generates a boxplot for every condition'''
        assert level in self.levels
        data = []
        conditions = sorted(self.conditions)
        for condition in conditions:
            data.append([value_in_nested_dict(d, self.path_to_metric)
                         for c, _i, l, d in self.data
                         if c == condition and l == level])
        plt.boxplot(data)
        plt.scatter(range(1, len(conditions) + 1), [np.mean(x) for x in data])
        plt.xticks(range(1, len(conditions) + 1), conditions)
        plt.show()

    def plot_iterations(self, iterations=None, conditions=None, levels=None):
        '''generates a plot with iterations on the x axis, results in the set
           score on the y axis for given conditions and levels'''
        # check given iterations
        if iterations is None:
            iterations = self.iterations
        else:
            iterations = [x for x in iterations if x in self.iterations]
        # check given conditions
        if conditions is None:
            conditions = self.conditions
        else:
            conditions = [x for x in conditions if x in self.conditions]
        # check given levels
        if levels is None:
            levels = self.levels
        else:
            levels = [x for x in levels if x in self.levels]
        # possible colored marks
        marker_colors = list('bgecmyk')
        marker_types = list('ov^<>sp*+xd')
        marker = [c + t for c, t in product(marker_colors, marker_types)]
        shuffle(marker)
        # generate plot
        groups = product(conditions, levels)
        for _nr, (condition, level) in enumerate(groups):
            # get data
            y = [value_in_nested_dict(d, self.path_to_metric)
                 for c, i, l, d in self.data
                 if c == condition and l == level and i in iterations]
            # plot data
            plt.plot(range(len(iterations)), y, '-',
                     label=condition + '-' + level)  # marker[nr]
        plt.legend(shadow=True, fancybox=True)
        plt.show()

    def save(self, filename):
        '''dumps collected results as a gzip pickle file'''
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f, -1)
