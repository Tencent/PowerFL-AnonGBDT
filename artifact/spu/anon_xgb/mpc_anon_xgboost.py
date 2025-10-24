import argparse
import copy
import json
import math
import multiprocessing
import sys
from statistics import mode
import time

import jax.lax
import multiprocess
import spu.spu_pb2
from sklearn.metrics import roc_auc_score

from typing import Any, Dict, List, Tuple
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functools import reduce

import spu.utils.distributed as ppd
from spu.utils.distributed import PYU, SPU

from cuckoo_hash import CuckooHasher
from mpc_xgboost import load_dataset_by_config, XgbModel, ppd_init, load_feature_r1, load_feature_r2


def preprocess(x, y=None, eps=1.27):
    r = 1
    if y is not None:
        r = 0

    native_cuckoo_map = CuckooHasher(r)
    remote_cuckoo_map = CuckooHasher(1 - r)

    nbin = math.ceil(x.shape[0] * eps)

    native_bin_pos, native_bin_value = native_cuckoo_map.hash_cuckoo(
        x[:, 0].astype(np.uint64), nbin
    )
    remote_bin_pos_arr, remote_bin_value_arr = remote_cuckoo_map.hash_simple(
        x[:, 0].astype(np.uint64), nbin
    )

    cols = x.shape[1] - 1
    dataset = np.zeros((nbin, cols), dtype=float)
    for i in range(nbin):
        idx = native_bin_pos[i]
        if idx != -1:
            dataset[i] = x[idx][1:].copy()

    label = None
    if y is not None:
        label = np.ndarray((nbin,))
        for i in range(nbin):
            idx = native_bin_pos[i]
            if idx != -1:
                label[i] = y[idx]
            else:
                label[i] = 0

    return native_bin_pos, native_bin_value, remote_bin_pos_arr, remote_bin_value_arr, dataset, label

def preprocess_naive(x, y=None, eps=1.27):
    n = len(x)

    cuckoo_map = CuckooHasher(1)
    nbin = math.ceil(x.shape[0] * eps)

    native_bin_pos, native_bin_value = cuckoo_map.hash_cuckoo(
        x[:, 0].astype(np.uint64), nbin
    )
    remote_bin_pos_arr, remote_bin_value_arr = cuckoo_map.hash_simple(
        x[:, 0].astype(np.uint64), nbin
    )

    cols = x.shape[1] - 1
    dataset = np.zeros((nbin, cols), dtype=float)
    for i in range(nbin):
        idx = native_bin_pos[i]
        if idx != -1:
            dataset[i] = x[idx][1:].copy()

    label = None
    if y is not None:
        label = np.ndarray((nbin,))
        for i in range(nbin):
            idx = native_bin_pos[i]
            if idx != -1:
                label[i] = y[idx]
            else:
                label[i] = 0

    return native_bin_pos, native_bin_value, remote_bin_pos_arr, remote_bin_value_arr, dataset, label

def compute_cuckoo(x: np.ndarray, r: int, eps=1.27):
    cuckoo_map = CuckooHasher(r)
    remote_cuckoo_map = CuckooHasher(1 - r)

    nbin = math.ceil(x.shape[0] * eps)
    native_bin_pos, native_bin_value = cuckoo_map.hash_cuckoo(
        x[:, 0].astype(np.uint64), nbin
    )
    remote_bin_pos_arr, remote_bin_value_arr = remote_cuckoo_map.hash_simple(
        x[:, 0].astype(np.uint64), nbin
    )

    return native_bin_pos, native_bin_value, remote_bin_pos_arr, remote_bin_value_arr

def build_maps(context: Dict[str, Any], x: np.ndarray):
    buckets_map = np.zeros(
        (x.shape[0], x.shape[1] * context['buckets']), dtype=np.bool_
    )
    order_map = np.zeros((x.shape[0], x.shape[1]), dtype=np.int8)
    split_points = list()

    for f in range(x.shape[1]):
        bins, split_point = pd.qcut(x[:, f], context['buckets'], labels=False, duplicates='drop', retbins=True)
        order_map[:, f] = bins
        sum_bin = None
        for b in range(split_point.size - 1):
            bin = np.flatnonzero(bins == b)
            if sum_bin is None:
                sum_bin = bin
            else:
                sum_bin = np.concatenate((sum_bin, bin), axis=None)
            buckets_map[sum_bin, f * context['buckets'] + b] = 1
        split_points.append(list(np.delete(split_point, (0,))))
    return buckets_map, order_map, split_points

def pyu_global_setup(x: np.ndarray, buckets: int, r: int, eps=1.27):
    context = dict()
    context['buckets'] = buckets

    xx = np.delete(x, 0, axis=1).astype(np.float64)
    buckets_map, order_map, split_points = build_maps(context, xx)

    bin_pos, bin_value, bin_pos_arr, bin_value_arr = compute_cuckoo(x, r, eps)
    nbin = len(bin_pos)
    cols = x.shape[1] - 1

    new_buckets_map = np.zeros(
        (nbin, cols * context['buckets']), dtype=np.bool_
    )
    new_order_map = np.zeros((nbin, cols), dtype=np.int8)

    for i in range(len(bin_pos)):
        idx = bin_pos[i]
        if idx != -1:
            new_buckets_map[i] = buckets_map[idx]
            new_order_map[i] = order_map[idx]

    context['order_map'] = new_order_map
    context['split_points'] = split_points

    context['bin_pos'] = np.array(bin_pos)
    context['bin_pos_arr'] = [arr for arr in bin_pos_arr]
    bin_len = np.array([len(arr) for arr in bin_pos_arr])

    bin_value_flatten = list()
    for arr in bin_value_arr:
        bin_value_flatten.extend(arr)

    return new_buckets_map, np.array(bin_value), np.array(bin_value_flatten), bin_len, context

def spu_global_setup(bin_value_list: List[np.ndarray], bin_value_flatten: List[np.ndarray],
                     bin_len_list: List[np.ndarray], buckets_list: List[np.ndarray]) -> Dict[str, Any]:
    context = dict()
    fed_num = len(bin_value_list)
    for r in range(fed_num):
        name_native_mask = 'native_mask' + str(r)
        context[name_native_mask] = jnp.isin(bin_value_list[r], bin_value_flatten[1-r])

        name_remote_mask = 'remote_mask' + str(r)
        context[name_remote_mask] = jnp.isin(bin_value_flatten[r], bin_value_list[1-r])

    for r in range(fed_num):
        name_bin_len = 'bin_len' + str(r)
        context[name_bin_len] = bin_len_list[r]

        name_buckets_map = 'buckets_map' + str(r)
        context[name_buckets_map] = buckets_list[r]

    return context

def pyu_bin_locate(context: Dict[str, Any], y: np.ndarray):
    bin_pos = context['bin_pos']
    new_y = np.zeros(len(bin_pos), dtype=np.int64)
    for i in range(len(bin_pos)):
        idx = bin_pos[i]
        if idx != -1:
            new_y[i] = ppd.get(y)[idx]
    return new_y

def pyu_bin_transform(context: Dict[str, Any], y: np.ndarray):
    bin_pos = context['bin_pos']
    bin_pos_arr = context['bin_pos_arr']

    bin_pos_index = jnp.where(bin_pos != -1)[0]
    prime_pos_index = ppd.get(bin_pos)[bin_pos_index]
    yprime = jnp.array([0] * len(prime_pos_index))
    yprime = yprime.at[prime_pos_index].set(ppd.get(y)[bin_pos_index])

    # for i in range(len(native_bin_pos)):
    #     idx = native_bin_pos[i]
    #     if jnp.not_equal(idx, -1):
    #         yprime[idx] = y[i]

    yremote_arr = list()
    for arr in ppd.get(bin_pos_arr):
        for idx in arr:
            if idx != -1:
                yremote_arr.append(yprime[idx])
            else:
                yremote_arr.append(0)
    yremote_arr = np.array(yremote_arr)

    return yremote_arr

def spu_label_init(context: Dict[str, Any], y0: np.ndarray, y1: np.ndarray):
    r = 0
    name_native_mask = 'native_mask' + str(r)
    native_bools = context[name_native_mask]

    name_remote_mask = 'remote_mask' + str(r)
    remote_bools = context[name_remote_mask]

    name_bin_len = 'bin_len' + str(r)
    bin_len = context[name_bin_len]

    # print("in spu_label_init: ", bools.tolist())
    # print("in spu label init, bool_count = ", jnp.count_nonzero(bools))

    name_y0 = 'y' + str(r)
    name_y1 = 'y' + str(1 - r)

    t0 = jnp.multiply(native_bools, y0)
    tt = jnp.multiply(remote_bools, y1)
    index = jnp.arange(len(tt))

    t1 = list()
    offset = 0
    for ll in bin_len:
        bools = jnp.array([jnp.logical_and(offset <= index, index < (offset + ll))])
        vv = jnp.sum(jnp.multiply(bools, tt))
        t1.append(vv)
        offset = offset + ll
    t1 = jnp.array(t1)

    context[name_y0] = t0
    context[name_y1] = t1

    # bin_pos0 = context['native_bin_pos' + str(0)]
    # tprime0 = [0] * n
    # for i in range(len(bin_pos0)):
    #     idx = bin_pos0[i]
    #     if idx != -1:
    #         tprime0[idx] = t0[i]
    # tprime0 = np.array(tprime0)
    #
    # bin_pos1 = context['native_bin_pos' + str(1)]
    # tprime1 = [0] * n
    # for i in range(len(bin_pos1)):
    #     idx = bin_pos1[i]
    #     if idx != -1:
    #         tprime1[idx] = t1[i]
    # tprime1 = np.array(tprime1)
    # assert np.array_equal(tprime0, tprime1)

    return context

def predict_weight_select(x: np.ndarray, tree: XgbModel.XgbTree):
    split_nodes = len(tree.split_features)
    select = np.zeros((x.shape[0], split_nodes + 1), dtype=np.int8)
    for r in range(x.shape[0]):
        row = x[r, :]
        idxs = list()
        idxs.append(0)
        while len(idxs):
            idx = idxs.pop(0)
            if idx < split_nodes:
                f = tree.split_features[idx]
                v = tree.split_values[idx]
                if f == -1:
                    idxs.append(idx * 2 + 1)
                    idxs.append(idx * 2 + 2)
                else:
                    if row[f] <= v:
                        idxs.append(idx * 2 + 1)
                    else:
                        idxs.append(idx * 2 + 2)
            else:
                leaf_idx = idx - split_nodes
                select[r, leaf_idx] = 1
    return select

def spu_predict_tree_weight(selects: List[np.ndarray], weights: np.ndarray):
    select = selects[0]
    for i in range(1, len(selects)):
        select = select * selects[i]
    assert (select.shape[1] == weights.shape[0]), f"select {select.shape}, weights {weights.shape}"
    return jnp.matmul(select, weights)

def sigmoid(x):
    return 0.5 * (x / jnp.sqrt(1 + jnp.power(x, 2))) + 0.5

def compute_gh(y: np.ndarray, pred: np.ndarray):
    yhat = sigmoid(pred)
    g = yhat - y
    h = yhat * (1 - yhat)
    return g, h

def compute_obj(G: np.ndarray, H: np.ndarray, _lambda=0.5):
    return G * (G / (H + _lambda))

def spu_tree_setup(context: Dict[str, Any], pred_list, fed_num: int) -> Dict[str, Any]:
    for r in range(fed_num):
        name_root_s = 'root_s' + str(r)
        if name_root_s not in context:
            name_native_bin_value = 'native_mask' + str(r)
            native_mask = context[name_native_bin_value]
            context[name_root_s] = native_mask.reshape(1, native_mask.shape[0]).astype(dtype=np.int8)

        name_y = 'y' + str(r)
        gh = compute_gh(context[name_y], pred_list[r])

        name_g = 'g' + str(r)
        context[name_g] = jnp.multiply(context[name_root_s], gh[0])

        name_h = 'h' + str(r)
        context[name_h] = jnp.multiply(context[name_root_s], gh[1])

    return context

def pyu_tree_setup(context: Dict[str, Any]):
    context['tree'] = XgbModel.XgbTree()
    return context

def init_pred(base: float, samples: int):
    shape = (1, samples)
    return jnp.full(shape, base)

def spu_find_best_split_bucket(context: Dict[str, Any], nodes_slist, last_level: bool):
    gain_list = list()
    for r in range(len(nodes_slist)):
        nodes_s = nodes_slist[r]
        l_nodes_s = [nodes_s[idx] for idx in range(len(nodes_s)) if idx % 2 == 0]

        name_cache = 'cache' + str(r)
        if name_cache in context:
            GL_cache = context[name_cache][0]
            HL_cache = context[name_cache][1]
            assert len(GL_cache) == len(l_nodes_s)
        else:
            GL_cache = None
            HL_cache = None

        level_GL = list()
        level_HL = list()

        name_g = 'g' + str(r)
        name_h = 'h' + str(r)
        name_buckets_map = 'buckets_map' + str(r)
        for idx in range(len(l_nodes_s)):
            sg = jnp.multiply(context[name_g], l_nodes_s[idx])
            sh = jnp.multiply(context[name_h], l_nodes_s[idx])
            lchild_GL = jnp.matmul(sg, context[name_buckets_map])
            lchild_HL = jnp.matmul(sh, context[name_buckets_map])
            level_GL.append(lchild_GL)
            level_HL.append(lchild_HL)

            if GL_cache is not None:
                level_GL.append(GL_cache[idx] - lchild_GL)
                level_HL.append(HL_cache[idx] - lchild_HL)

        if not last_level:
            context[name_cache] = (level_GL, level_HL)
        elif name_cache in context:
            del context[name_cache]

        GL = jnp.concatenate(level_GL, axis=0)
        HL = jnp.concatenate(level_HL, axis=0)

        GR = GL[:, -1].reshape(-1, 1) - GL
        HR = HL[:, -1].reshape(-1, 1) - HL

        obj_l = compute_obj(GL, HL)
        obj_r = compute_obj(GR, HR)

        obj = obj_l[:, -1].reshape(-1, 1)
        gain = obj_l + obj_r - obj

        gain_list.append(gain)
    gains = jnp.concatenate(gain_list, axis=1)
    split_buckets = jnp.argmax(gains, axis=1)
    return split_buckets, context

def do_split(context: Dict[str, Any], split_bucket: int):
    name_tree = 'tree'
    if split_bucket == -1:
        context[name_tree].insert_split_node(-1, float("inf"))
        return context
    else:
        feature = int(split_bucket / context['buckets'])
        split_point_idx = split_bucket % context['buckets']
        context[name_tree].insert_split_node(
            feature, context['split_points'][feature][split_point_idx]
        )
        ls = (context['order_map'][:, feature] <= split_point_idx).astype(np.int8).reshape(1, context['order_map'].shape[0])

    return ls[0], context

def spu_synchronize(context: Dict[str, Any], ls: np.ndarray, r: int):
    name_remote_mask = 'remote_mask' + str(r)
    remote_bools = context[name_remote_mask]

    name_bin_len = 'bin_len' + str(r)
    bin_len = context[name_bin_len]

    # print("in spu_label_init: ", bools.tolist())
    # print("in spu label init, bool_count = ", jnp.count_nonzero(bools))

    tt = jnp.multiply(remote_bools, ls)
    index = jnp.arange(len(tt))

    yy = list()
    offset = 0
    for ll in bin_len:
        bools = jnp.array([jnp.logical_and(offset <= index, index < (offset + ll))])
        vv = jnp.sum(jnp.multiply(bools, tt))
        yy.append(vv)
        offset = offset + ll
    yy = jnp.array(yy)

    return yy

def tree_eval(nodes_s: List[np.ndarray], weight: np.ndarray):
    nodes = jnp.concatenate(nodes_s, axis=0)
    return jnp.matmul(weight, nodes)

def get_child_select(nodes_s: np.ndarray, lchilds_s: np.ndarray):
    childs_s = list()
    for current, lchild in zip(nodes_s, lchilds_s):
        ls = current * lchild
        rs = current - ls
        childs_s.append(ls)
        childs_s.append(rs)
    return childs_s

def compute_weight(G: float, H: float, _lambda=0.5):
    return -((G / (H + _lambda)) * 0.3)

def get_weight(context: Dict[str, Any], s: np.ndarray, r:int):
    name_g = 'g' + str(r)
    name_h = 'h' + str(r)

    g_sum = (context[name_g] * s).sum(axis=1)
    h_sum = (context[name_h] * s).sum(axis=1)
    return compute_weight(g_sum, h_sum)

def do_leaf(context: Dict[str, Any], ss, r=0):
    s = jnp.concatenate(ss, axis=0)
    return get_weight(context, s, r)

def tree_finish(context: Dict[str, Any]) -> XgbModel.XgbTree:
    name_tree = 'tree'
    return context[name_tree]


class MockAnonXgb():
    def __init__(self, eps=1.27) -> None:
        self.eps = eps

    def train(self, trees: int, depth: int, buckets: int, dataset, y):
        self.trees = trees
        self.depth = depth
        self.buckets = buckets
        self.n = dataset[0].shape[0]

        buckets_list = list()
        self.buckets_size = list()
        self.fed_num = len(dataset)
        self.pyus_context = list()
        self.spu_context = dict()
        self.nbin_size = list()
        bin_value_list = list()
        bin_value_flatten_list = list()
        bin_len_list = list()

        for r in range(self.fed_num):
            mbools, bin_value, bin_value_flatten, bin_len, context = pyu_global_setup(
                dataset[r], self.buckets, r, self.eps)

            buckets_list.append(mbools)
            bin_value_list.append(bin_value)
            bin_value_flatten_list.append(bin_value_flatten)
            bin_len_list.append(bin_len)

            self.nbin_size.append(mbools.shape[0])
            self.buckets_size.append(mbools.shape[1])
            self.pyus_context.append(context)

        # for r in range(self.fed_num):
        #     bin_value_list.append(self.pyus_context[r]['bin_value'])
        #     bin_value_flatten_list.append(self.pyus_context[r]['bin_value_flatten'])
        #     bin_len_list.append(self.pyus_context[r]['bin_len'])

        # for r in range(self.fed_num):
        #     native_mask = self.spu(lambda x, y: jnp.isin(x, y))(self.pyus_context[r]['bin_value'],
        #                                                         self.pyus_context[1-r]['bin_value_flatten'])
        #     print(f"native_mask[{r}]: ", jnp.count_nonzero(ppd.get(native_mask)))

        self.spu_context = spu_global_setup(bin_value_list, bin_value_flatten_list,
                                            bin_len_list, buckets_list)
        del bin_value_list, bin_value_flatten_list, bin_len_list, buckets_list

        label = pyu_bin_locate(self.pyus_context[0], y)
        label_ = pyu_bin_transform(self.pyus_context[0], label)
        self.spu_context = spu_label_init(self.spu_context, label, label_)

        tree_num = 0
        pred_list = list()
        for r in range(self.fed_num):
            pred = init_pred(0, self.nbin_size[r])
            pred_list.append(pred)

        model = XgbModel()
        while tree_num < self.trees:
            for r in range(self.fed_num):
                self.pyus_context[r] = pyu_tree_setup(self.pyus_context[r])
            self.spu_context = spu_tree_setup(self.spu_context, pred_list, self.fed_num)

            tree, nodes_slist, weight = self.train_tree()
            model.trees.append(tree)
            model.weights.append(weight)

            print(f"in {tree_num}-th tree, leaf weights: {weight}")

            tree_num = tree_num + 1
            if tree_num < self.trees:
                for r in range(self.fed_num):
                    current = tree_eval(nodes_slist[r], weight)
                    pred_list[r] = (lambda x, y: x+y)(pred_list[r], current)

        return model

    def train_tree(self):
        nodes_slist = list()
        for r in range(self.fed_num):
            name_root_s = 'root_s' + str(r)
            root_s = self.spu_context[name_root_s]
            nodes_s = [root_s]
            print(f"roots_[{r}]: ", jnp.count_nonzero(ppd.get(root_s)))
            nodes_slist.append(nodes_s)

        # print("root_slist[0]: ", len(nodes_slist[0]), " ", nodes_slist[0][0].shape)

        weight = None
        for level in range(self.depth + 1):
            if level < self.depth:
                nodes_slist = self.train_level(nodes_slist, level)
                print(f"in level {level}: nodes_slist: ", len(nodes_slist))
                # print(f"in level {level}, nodes_slist[0]: {len(nodes_slist[0])}, {nodes_slist[0][0].shape}")
            else:
                weight = do_leaf(self.spu_context, nodes_slist[0], 0)

                # test weight equality
                # weight0 = do_leaf(self.fed_context, nodes_slist[0], 0)
                # weight1 = do_leaf(self.fed_context, nodes_slist[1], 1)
                # assert np.allclose(weight0, weight1), "weight not equal"

        tree = [
            tree_finish(self.pyus_context[r])
            for r in range(self.fed_num)
        ]

        return tree, nodes_slist, weight

    def _split_rank(self, split_bucket: int):
        pre_end_pos = 0
        for r in range(len(self.buckets_size)):
            current_end_pos = pre_end_pos + self.buckets_size[r]
            if split_bucket < current_end_pos:
                return r, split_bucket - pre_end_pos
            pre_end_pos += self.buckets_size[r]

        assert False, "should not be here"

    def train_level(self, nodes_slist, level):
        last_level = level == (self.depth - 1)
        spu_split_buckets, self.spu_context = spu_find_best_split_bucket(self.spu_context, nodes_slist, last_level)

        split_buckets = list(ppd.get(spu_split_buckets))
        assert len(split_buckets) == len(nodes_slist[0])

        # print(f"in level {level}, split_buckets: ", split_buckets)

        lchilds_slist = [list() for _ in range(self.fed_num)]
        for s in split_buckets:
            split_rank, split_rank_idx = self._split_rank(s)
            for r in range(self.fed_num):
                if r == split_rank:
                    lchild_s, self.pyus_context[r] = do_split(self.pyus_context[r], split_rank_idx)
                    lchild_s_ = pyu_bin_transform(self.pyus_context[r], lchild_s)
                    remote_lchild_s = spu_synchronize(self.spu_context, lchild_s_, r)

                    if r == 0:
                        lchilds_slist[r].append(lchild_s)
                        lchilds_slist[1-r].append(remote_lchild_s)
                    else:
                        lchilds_slist[1-r].append(remote_lchild_s)
                        lchilds_slist[r].append(lchild_s)
                else:
                    self.pyus_context[r] = do_split(self.pyus_context[r], -1)

        # lchilds_slist = [np.array(xx) for xx in lchilds_slist]
        assert len(lchilds_slist[0]) == len(split_buckets)
        childs_slist = list()
        for r in range(self.fed_num):
            child_s = get_child_select(nodes_slist[r], lchilds_slist[r])
            childs_slist.append(child_s)

        del lchilds_slist, split_buckets
        return childs_slist

    def _update_predict_tree(self, pred, dataset, tree: List[XgbModel.XgbTree], weight):
        assert len(tree) == len(dataset)

        weight_selects = list()
        for idx in range(len(dataset)):
            ws = predict_weight_select(dataset[idx], tree[idx])
            weight_selects.append(ws)

        current = spu_predict_tree_weight(weight_selects, weight)

        if pred is not None:
            return (lambda x, y: x + y)(pred, current)
        else:
            return current

    def predict(self, dataset, model: XgbModel):
        if len(model.trees[0]) == 0:
            return None
        assert len(dataset) == len(model.trees[0])
        pred = None
        for idx in range(len(model.trees)):
            pred = self._update_predict_tree(pred, dataset, model.trees[idx], model.weights[idx])

        return sigmoid(pred)

class MPCAnonXgb():
    def __init__(self, spu: SPU, eps=1.27) -> None:
        self.spu = spu
        self.eps = eps

    def train(self, trees: int, depth: int, buckets: int, dataset: List[PYU.Object], y: PYU.Object):
        self.trees = trees
        self.depth = depth
        self.buckets = buckets
        self.pyus = [data.device for data in dataset]
        self.n = dataset[0].shape[0]

        buckets_list = list()
        self.buckets_size = list()
        self.fed_num = len(self.pyus)
        self.pyus_context = list()
        self.spu_context = dict()
        self.nbin_size = list()
        bin_value_list = list()
        bin_value_flatten_list = list()
        bin_len_list = list()

        for r in range(self.fed_num):
            mbools, bin_value, bin_value_flatten, bin_len, context = self.pyus[r](
                pyu_global_setup)(dataset[r], self.buckets, r, self.eps)

            print("mbools: ", type(mbools), " ", type(bin_value), " ", type(bin_value_flatten), " ", type(bin_len))

            buckets_list.append(mbools)
            bin_value_list.append(bin_value)
            bin_value_flatten_list.append(bin_value_flatten)
            bin_len_list.append(bin_len)

            self.nbin_size.append(mbools.shape[0])
            self.buckets_size.append(mbools.shape[1])
            self.pyus_context.append(context)

        # for r in range(self.fed_num):
        #     bin_value_list.append(self.pyus_context[r]['bin_value'])
        #     bin_value_flatten_list.append(self.pyus_context[r]['bin_value_flatten'])
        #     bin_len_list.append(self.pyus_context[r]['bin_len'])

        # for r in range(self.fed_num):
        #     native_mask = self.spu(lambda x, y: jnp.isin(x, y))(self.pyus_context[r]['bin_value'],
        #                                                         self.pyus_context[1-r]['bin_value_flatten'])
        #     print(f"native_mask[{r}]: ", jnp.count_nonzero(ppd.get(native_mask)))

        self.spu_context = self.spu(spu_global_setup)(bin_value_list, bin_value_flatten_list,
                                                      bin_len_list, buckets_list)
        del bin_value_list, bin_value_flatten_list, bin_len_list, buckets_list

        label = self.pyus[0](pyu_bin_locate)(self.pyus_context[0], y)
        label_ = self.pyus[0](pyu_bin_transform)(self.pyus_context[0], label)
        self.spu_context = self.spu(spu_label_init)(self.spu_context, label, label_)

        tree_num = 0
        pred_list = list()
        for r in range(self.fed_num):
            pred = self.spu(init_pred, static_argnums=(0, 1))(0, self.nbin_size[r])
            pred_list.append(pred)

        model = XgbModel()
        while tree_num < self.trees:
            for r in range(self.fed_num):
                self.pyus_context[r] = self.pyus[r](pyu_tree_setup)(self.pyus_context[r])
            self.spu_context = self.spu(spu_tree_setup, static_argnums=(2, ))(self.spu_context, pred_list, self.fed_num)

            tree, nodes_slist, weight = self.train_tree()
            model.trees.append(tree)
            model.weights.append(weight)

            print(f"in {tree_num}-th tree, leaf weights: {ppd.get(weight)}")

            tree_num = tree_num + 1
            if tree_num < self.trees:
                for r in range(self.fed_num):
                    current = self.spu(tree_eval)(nodes_slist[r], weight)
                    pred_list[r] = self.spu(lambda x, y: x+y)(pred_list[r], current)

        return model

    def train_tree(self):
        nodes_slist = list()
        for r in range(self.fed_num):
            name_root_s = 'root_s' + str(r)
            root_s = self.spu_context[name_root_s]
            nodes_s = [root_s]
            print(f"roots_[{r}]: ", jnp.count_nonzero(ppd.get(root_s)))
            nodes_slist.append(nodes_s)

        # print("root_slist[0]: ", len(nodes_slist[0]), " ", nodes_slist[0][0].shape)

        weight = None
        for level in range(self.depth + 1):
            if level < self.depth:
                nodes_slist = self.train_level(nodes_slist, level)
                print(f"in level {level}: nodes_slist: ", len(nodes_slist))
                # print(f"in level {level}, nodes_slist[0]: {len(nodes_slist[0])}, {nodes_slist[0][0].shape}")
            else:
                weight = self.spu(do_leaf, static_argnums=(2,))(self.spu_context, nodes_slist[0], 0)

                # test weight equality
                # weight0 = do_leaf(self.fed_context, nodes_slist[0], 0)
                # weight1 = do_leaf(self.fed_context, nodes_slist[1], 1)
                # assert np.allclose(weight0, weight1), "weight not equal"

        tree = [
            self.pyus[r](tree_finish)(self.pyus_context[r])
            for r in range(self.fed_num)
        ]

        return tree, nodes_slist, weight

    def _split_rank(self, split_bucket: int):
        pre_end_pos = 0
        for r in range(len(self.buckets_size)):
            current_end_pos = pre_end_pos + self.buckets_size[r]
            if split_bucket < current_end_pos:
                return r, split_bucket - pre_end_pos
            pre_end_pos += self.buckets_size[r]

        assert False, "should not be here"

    def train_level(self, nodes_slist, level):
        last_level = level == (self.depth - 1)
        spu_split_buckets, self.spu_context = self.spu(
            spu_find_best_split_bucket, static_argnums=(2,))(self.spu_context, nodes_slist, last_level)

        split_buckets = list(ppd.get(spu_split_buckets))
        assert len(split_buckets) == len(nodes_slist[0])

        # print(f"in level {level}, split_buckets: ", split_buckets)

        lchilds_slist = [list() for _ in range(self.fed_num)]
        for s in split_buckets:
            split_rank, split_rank_idx = self._split_rank(s)
            for r in range(self.fed_num):
                if r == split_rank:
                    lchild_s, self.pyus_context[r] = self.pyus[r](do_split)(self.pyus_context[r], split_rank_idx)
                    lchild_s_ = self.pyus[r](pyu_bin_transform)(self.pyus_context[r], lchild_s)
                    remote_lchild_s = self.spu(spu_synchronize, static_argnums=(2, ))(self.spu_context, lchild_s_, r)

                    if r == 0:
                        lchilds_slist[r].append(lchild_s)
                        lchilds_slist[1-r].append(remote_lchild_s)
                    else:
                        lchilds_slist[1-r].append(remote_lchild_s)
                        lchilds_slist[r].append(lchild_s)
                else:
                    self.pyus_context[r] = self.pyus[r](do_split)(self.pyus_context[r], -1)

        # lchilds_slist = [np.array(xx) for xx in lchilds_slist]
        assert len(lchilds_slist[0]) == len(split_buckets)
        childs_slist = list()
        for r in range(self.fed_num):
            child_s = self.spu(get_child_select)(nodes_slist[r], lchilds_slist[r])
            childs_slist.append(child_s)

        del lchilds_slist, split_buckets
        return childs_slist

    def _update_predict_tree(self, pred: SPU.Object, dataset: List[PYU.Object], tree: List[XgbModel.XgbTree],
                             weight: SPU.Object):
        assert len(tree) == len(dataset)

        weight_selects = list()
        for idx in range(len(dataset)):
            assert tree[idx].device == dataset[idx].device
            ws = dataset[idx].device(predict_weight_select)(dataset[idx], tree[idx])
            weight_selects.append(ws)

        current = self.spu(spu_predict_tree_weight)(weight_selects, weight)

        if pred is not None:
            return self.spu(lambda x, y: x + y)(pred, current)
        else:
            return current

    def predict(self, dataset: List[PYU.Object], model: XgbModel) -> SPU.Object:
        if len(model.trees[0]) == 0:
            return None
        assert len(dataset) == len(model.trees[0])
        pred = None
        for idx in range(len(model.trees)):
            pred = self._update_predict_tree(pred, dataset, model.trees[idx], model.weights[idx])

        return self.spu(sigmoid)(pred)

def gen_data(path, dataset_config=None):
    if dataset_config is None:
        with open(path, "r") as f:
            dataset_config = json.load(f)

    x0, x1, y = load_dataset_by_config(dataset_config)
    x1 = x1[:, :x1.shape[1] - 5]

    n = x0.shape[0]
    mask64 = np.uint64(0xffffffffffffffff)

    id0 = np.random.randint(0, mask64, n, dtype=np.uint64)
    id1 = np.random.randint(0, mask64, n, dtype=np.uint64)
    for i in range(n):
        id1[i] = id0[i]
        # if np.random.randint(0, 4, dtype=np.int8) == 0:
        #     id1[i] = id0[i]

    id0 = id0.reshape(n, 1).astype('O')
    id1 = id1.reshape(n, 1).astype('O')

    x0 = x0.astype('O')
    x1 = x1.astype('O')

    x0 = np.concatenate([id0, x0], axis=1)
    x1 = np.concatenate([id1, x1], axis=1)

    return x0, x1, y

def test_mock_anon_xgb():
    prime_x0, prime_x1, prime_y = gen_data("ds_breast_cancer_basic.json")

    # n = prime_x0.shape[0]
    #
    # bin_pos0, bin_value0, bin_pos_arr0, bin_value_arr0, x0, y = preprocess_naive(prime_x0, prime_y)
    # bin_pos1, bin_value1, bin_pos_arr1, bin_value_arr1, x1, _ = preprocess_naive(prime_x1)
    #
    # native_bin_pos = [bin_pos0, bin_pos1]
    # native_bin_value = [bin_value0, bin_value1]
    # remote_bin_pos_arr = [bin_pos_arr0, bin_pos_arr1]
    # remote_bin_value_arr = [bin_value_arr0, bin_value_arr1]
    #
    # bools0 = [arr.count(x) > 0 for (x, arr) in zip(bin_value0, bin_value_arr1)]
    # bools1 = [arr.count(x) > 0 for (x, arr) in zip(bin_value1, bin_value_arr0)]
    #
    # print(bools0)
    # print(bools1)

    dataset = [prime_x0, prime_x1]

    anon_xgb = MockAnonXgb()
    model = anon_xgb.train(3, 3, 4, dataset, prime_y)

    x0_data = np.delete(prime_x0, 0, axis=1).astype(np.float64)
    x1_data = np.delete(prime_x1, 0, axis=1).astype(np.float64)
    prime_dataset = [x0_data, x1_data]
    yhat = anon_xgb.predict(prime_dataset, model)
    score = roc_auc_score(prime_y, yhat)
    print(f"auc {score}")

def test_mpc_anon_xgb():
    dataset_config = ppd_init()
    x1, x2, y = gen_data("ds_breast_cancer_basic.json", dataset_config=dataset_config)
    x1_data = np.delete(x1, 0, axis=1).astype(np.float64)
    x2_data = np.delete(x2, 0, axis=1).astype(np.float64)

    x1, y = ppd.device("P1")(load_feature_r1)(x1, y)
    x2 = ppd.device("P2")(load_feature_r2)(x2)

    dataset = [x1, x2]

    start = time.time()
    mpc_anon_xgb = MPCAnonXgb(ppd.device("SPU"))
    model = mpc_anon_xgb.train(3, 3, 4, dataset, y)
    train_time = time.time() - start
    print(f"train time {train_time}")

    x1_data = ppd.device("P1")(load_feature_r2)(x1_data)
    x2_data = ppd.device("P2")(load_feature_r2)(x2_data)
    dataset_without_id = [x1_data, x2_data]

    start = time.time()
    yhat = ppd.get(mpc_anon_xgb.predict(dataset_without_id, model))
    predict_time = time.time() - start
    print(f"predict time {predict_time}")

    score = roc_auc_score(ppd.get(y), yhat)
    print(f"auc {score}")

if __name__ == "__main__":
    # test_mock_anon_xgb()
    test_mpc_anon_xgb()

    # in 0 - th
    # tree, leaf
    # weights: [0.5596331   0.3         0.42857146 - 0.25714287  0.20000002 - 0.3
    #           - 0.20000002 - 0.5853659]
    # in 1 - th
    # tree, leaf
    # weights: [0.35922807 - 0.00123364  0.26314858 - 0.309433    0.27809232 - 0.28034157
    #           - 0. - 0.38993555]
    # in 2 - th
    # tree, leaf
    # weights: [0.32979894 - 0.003898    0.3015687 - 0.30262536  0.38602686 - 0.08060722
    #           - 0.06975597 - 0.34102154]

