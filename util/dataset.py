#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
from bidict import bidict
from scipy.sparse import csr_matrix
from copy import deepcopy


class dataset_csr(object):
    def __init__(self, pathfile, item_id='id', class_id='class',
                 group_func=lambda x: x.split(':', 1)[0]):
        # the arff attribute name identifying the item id attribute
        self.item_id = item_id
        # the arff attribute name identifying the class attribute
        self.class_id = class_id
        # a function extracting the group id from the item ids
        self.group_func = group_func
        # load arff file and make a dataframe for the features
        al = arff_loader(pathfile, item_id=item_id, class_id=class_id)
        self.X = al.dict['data']
        self.y = al.dict['targets']
        self.rows = al.dict['item_map']
        self.columns = al.dict['attribute_map']
        # attributes stores for each categorical feature the bimap from
        # indices to category names
        self.attributes = al.dict['attributes']
        # the target bimaps indices to class category names
        self.target_map = al.dict['target_map']
        del al

    def get_indices_for_groups(self, groups):
        rowmap = sorted([(i, s) for i, s in self.rows.iteritems()
                         if self.group_func(s) in groups])
        rows = [i for i, s in rowmap]
        return np.array(rows)

    def get_vectors_for_groups(self, groups, features=None, row_mapping=False, ignore_class=None):
        rowmap = sorted([(i, s) for i, s in self.rows.iteritems()
                         if (self.group_func(s) in groups and
                             self.target_map[self.y[i]] != ignore_class)])
        rows = [i for i, s in rowmap]
        if features is None:
            X = self.X[rows]
        else:
            X = self.X[rows]  # todo,features]
        y = self.y[rows]
        # eventually return the mapping from the output rows (the result of
        # the row slice) to the dataset rows
        if row_mapping:
            return_map = {k: v for k, v in enumerate(rowmap)}
            return X, y, return_map
        else:
            return X, y

    def get_group_names(self):
        return set(self.group_func(s) for _, s in self.rows.iteritems())

    def restrict_feature_set_to(self, feature_name_filter_func):
        r = deepcopy(self)
        mask = np.array([1 if feature_name_filter_func(r.columns[i]) else 0
                         for i in sorted(r.columns.keys())], dtype=bool)
        r.X = csr_matrix(r.X.todense()[:, mask])
        remaining_feature_names = [
            r.columns[i] for i, v in enumerate(mask) if v == 1]
        r.columns = bidict({
            i: v for i, v in enumerate(remaining_feature_names)})
        return r


class arff_loader(object):

    def __init__(self, pathfile, item_id='id', class_id='class'):
        self.dict = {'name': '',
                     'attributes': {},
                     'attribute_map': bidict({}),
                     'data': None,
                     'targets': [],
                     'target_names': [],
                     'target_map': bidict({}),
                     'item_names': [],
                     'item_map': bidict({})}
        self.arff_attribute_id_counter = 0
        self.matrix_attribute_id_counter = 0
        self.item_id = item_id
        self.item_id_nr = None
        self.class_id = class_id
        self.class_id_nr = None
        self.map_arff_aid_to_matrix_col = bidict({})
        self._load_sparse_arff(pathfile)

    def _set_relation(self, line):
        self.dict['name'] = line.split(' ')[1]

    def _register_attribute(self, line):
        arff_cnt = self.arff_attribute_id_counter
        self.arff_attribute_id_counter += 1

        _, quoted_name, rest = line.split(' ', 2)
        name = quoted_name.strip('\'')

        if name == self.class_id:
            self.class_id_nr = arff_cnt
            print " Class_id is arff feature %d." % self.class_id_nr
        elif name == self.item_id:
            self.item_id_nr = arff_cnt
            print " Item_id is arff feature %d." % self.item_id_nr
        else:
            # these features will be saved in the data matrix and get indexed
            self.dict['attribute_map'][self.matrix_attribute_id_counter] = name
            self.map_arff_aid_to_matrix_col[arff_cnt] = self.matrix_attribute_id_counter  # noqa
            self.matrix_attribute_id_counter += 1

        # attribute type
        if rest.startswith('numeric'):
            self.dict['attributes'][name] = None
        elif rest.startswith('string'):
            self.dict['attributes'][name] = None
        elif rest.startswith('{'):
            closing = rest.index('}')
            categories = [x.strip() for x in rest[1:closing].split(',')]
            if name == self.class_id:
                self.dict['target_names'] = categories
                self.dict['target_map'] = bidict({i: v for i, v in
                                                  enumerate(categories)})
            else:
                self.dict['attributes'][name] = bidict({i: v for i, v in
                                                        enumerate(categories)})
        else:
            print "Warning: unknown type of attribute %s: %s" % (name, rest)

    def _load_sparse_arff(self, pathfile):
        print "Loading arff file", pathfile
        s = codecs.open(pathfile, 'r', 'utf-8').read()
        lines = s.splitlines()
        data_starts_at = 0

        # read header
        for n, l in enumerate(lines):
            line = l.strip()
            if line.startswith('@relation'):
                self._set_relation(line)
            elif line.startswith('%'):
                # skip comments
                continue
            elif line.startswith('@attribute'):
                self._register_attribute(line)
            elif line.startswith('@data'):
                data_starts_at = n + 1
                break

        # read data matrix
        data_lines = lines[data_starts_at:]
        row = []
        col = []
        data = []
        nr_items = len(data_lines)
        nr_features = len(self.map_arff_aid_to_matrix_col)
        for i, line in enumerate(data_lines):
            fields = line.strip().strip('{}').split(',')
            for field in fields:
                f, v = field.strip().split(' ', 1)
                f = int(f)
                if f == self.class_id_nr:
                    self.dict['targets'].append(
                        self.dict['target_names'].index(v))
                elif f == self.item_id_nr:
                    self.dict['item_names'].append(v)
                    self.dict['item_map'][i] = v
                else:
                    row.append(i)
                    col.append(self.map_arff_aid_to_matrix_col[f])
                    try:
                        datum = float(v)
                    except ValueError:
                        m_id = self.map_arff_aid_to_matrix_col[f]
                        a_name = self.dict['attribute_map'][m_id]
                        datum = self.dict['attributes'][a_name][:v]
                    data.append(datum)
        self.dict['data'] = csr_matrix((data, (row, col)),
                                       shape=(nr_items, nr_features))
        self.dict['targets'] = np.array(self.dict['targets'])
        print "Done."
