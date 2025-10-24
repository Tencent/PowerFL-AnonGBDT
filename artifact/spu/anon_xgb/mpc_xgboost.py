import argparse
import json
import sys
import time

import multiprocess
from sklearn.metrics import roc_auc_score

from typing import Any, Dict, List, Tuple
import jax.numpy as jnp
import numpy as np
import pandas as pd

import spu.utils.distributed as ppd
from spu.utils.distributed import PYU, SPU

def mock_regression(n_samples, n_features, hardness=0.1, random_seed=None):
    """Generate a mock regression dataset.
    Use scikit learn make regression utils,
    which is much better than naively randomly sampled data.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression

    hardness should be between 0 and 1.
    1 -> completely random dataset
    0 -> completely clean dataset: no noisy feature and y values
    """
    from sklearn.datasets import make_regression

    hardness = max(min(hardness, 1.0), 0.0)
    informative_ratio = 1.0 - hardness
    X, y, coef = make_regression(
        n_samples,
        n_features,
        n_informative=int(informative_ratio * n_features),
        noise=hardness,
        coef=True,
        random_state=random_seed,
    )
    # coef is the underlying true coefficients for the linear model
    return X, y, coef

def mock_classification(n_samples, n_features, hardness=0.1, random_seed=None):
    """Generate a mock classification dataset.
    Use scikit learn make classification utils,
    which is much better than naively randomly sampled data.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification

    hardness should be between 0 and 1.
    1 -> completely random dataset
    0 -> completely clean dataset: no noisy feature/label
         and great separation between classes
    """
    from sklearn.datasets import make_classification

    hardness = max(min(hardness, 1.0), 0.0)
    informative_ratio = 1.0 - hardness
    redundant_ratio = (1.0 - informative_ratio) * hardness
    class_sep = 1.0 - hardness
    flip_y = 0.5 * hardness
    X, y = make_classification(
        n_samples,
        n_features,
        n_informative=int(informative_ratio * n_features),
        n_redundant=int(redundant_ratio * n_features),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_seed,
    )
    return X, y

def load_dataset_by_config(config):
    if config["use_mock_data"]:
        if config["problem_type"] == "regression":
            X, y, _ = mock_regression(
                config["n_samples"],
                config["n_features"],
                config["hardness"],
                config["random_seed"],
            )
        elif config["problem_type"] == "classification":
            X, y = mock_classification(
                config["n_samples"],
                config["n_features"],
                config["hardness"],
                config["random_seed"],
            )
    else:
        if config["builtin_dataset_name"] == "breast_cancer":
            from sklearn.datasets import load_breast_cancer

            ds = load_breast_cancer()
        elif config["builtin_dataset_name"] == "diabetes":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes()
        X, y = ds['data'], ds['target']
        # normalize, TODO(zoupeicheng.zpc): make preprocessing configurable
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
    split_index = int(X.shape[1] * config["left_slice_feature_ratio"])
    return X[:, :split_index], X[:, split_index:], y

def load_feature_r1(x, y):
    return x, y

def load_feature_r2(x):
    return x


def spu_distributed_up(nodes_def, devices_def):
    workers = []
    for node_id in nodes_def.keys():
        worker = multiprocess.Process(target=ppd.RPC.serve, args=(node_id, nodes_def))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

def ppd_init(config="3pc.json", dataset_config="ds_breast_cancer_basic.json"):
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default=config)
    parser.add_argument("-d", "--dataset_config", default=dataset_config)

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    with open(args.dataset_config, "r") as f:
        dataset_config = json.load(f)

    print(f"conf: {conf}")
    print(f"dataset_config: {dataset_config}")

    ppd.init(conf["nodes"], conf["devices"])
    return dataset_config


class XgbModel:
    def __init__(self) -> None:
        self.trees = list()
        self.weights = list()

    class XgbTree:
        def __init__(self) -> None:
            self.split_features = list()
            self.split_values = list()

        def insert_split_node(self, feature: int, value: float):
            assert isinstance(feature, int), f"feature {feature}"
            assert isinstance(value, float), f"value {value}"
            self.split_features.append(feature)
            self.split_values.append(value)

def sigmoid(x):
    return 0.5 * (x / jnp.sqrt(1 + jnp.power(x, 2))) + 0.5

def compute_obj(G: np.ndarray, H: np.ndarray, _lambda=0.5):
    return G * (G / (H + _lambda))

def compute_weight(G: float, H: float, _lambda=0.5):
    return -((G / (H + _lambda)) * 0.3)

def get_weight(context: Dict[str, Any], s: np.ndarray):
    g_sum = (context['g'] * s).sum(axis=1)
    h_sum = (context['h'] * s).sum(axis=1)
    return compute_weight(g_sum, h_sum)

def compute_gh(y: np.ndarray, pred: np.ndarray):
    yhat = sigmoid(pred)
    g = yhat - y
    h = yhat * (1 - yhat)
    return g, h

def spu_global_setup(buckets_map: List[np.ndarray], y: np.ndarray) -> Dict[str, Any]:
    context = dict()
    context['buckets_map'] = jnp.concatenate(buckets_map, axis=1)
    context['y'] = y

    return context

def spu_tree_setup(context: Dict[str, Any], pred: np.ndarray) -> Dict[str, Any]:
    gh = compute_gh(context['y'], pred)
    context['g'] = gh[0]
    context['h'] = gh[1]

    return context

def find_best_split_bucket(
        context: Dict[str, Any], l_nodes_s: List[np.ndarray], last_level: bool):
    if 'cache' in context:
        GL_cache = context['cache'][0]
        HL_cache = context['cache'][1]
        assert len(GL_cache) == len(l_nodes_s)
    else:
        GL_cache = None
        HL_cache = None

    level_GL = list()
    level_HL = list()

    for idx in range(len(l_nodes_s)):
        sg = context['g'] * l_nodes_s[idx]
        sh = context['h'] * l_nodes_s[idx]
        lchild_GL = jnp.matmul(sg, context['buckets_map'])
        lchild_HL = jnp.matmul(sh, context['buckets_map'])
        level_GL.append(lchild_GL)
        level_HL.append(lchild_HL)

        if GL_cache is not None:
            level_GL.append(GL_cache[idx] - lchild_GL)
            level_HL.append(HL_cache[idx] - lchild_HL)

    if not last_level:
        context['cache'] = (level_GL, level_HL)
    elif 'cache' in context:
        del context['cache']

    GL = jnp.concatenate(level_GL, axis=0)
    HL = jnp.concatenate(level_HL, axis=0)

    GR = GL[:, -1].reshape(-1, 1) - GL
    HR = HL[:, -1].reshape(-1, 1) - HL

    obj_l = compute_obj(GL, HL)
    obj_r = compute_obj(GR, HR)

    obj = obj_l[:, -1].reshape(-1, 1)

    gain = obj_l + obj_r - obj

    # print("gain: ", gain)

    split_buckets = jnp.argmax(gain, axis=1)

    return split_buckets, context

def init_pred(base: float, samples: int):
    shape = (1, samples)
    return jnp.full(shape, base)

def root_select(samples: int):
    return jnp.ones((1, samples))

def get_child_select(nodes_s: np.ndarray, lchilds_s: np.ndarray):
    childs_s = list()
    for current, lchild in zip(nodes_s, lchilds_s):
        ls = current * lchild
        rs = current - ls
        childs_s.append(ls)
        childs_s.append(rs)
    return childs_s

def predict_tree_weight(selects: List[np.ndarray], weights: np.ndarray):
    select = selects[0]
    for i in range(1, len(selects)):
        select = select * selects[i]
    assert (select.shape[1] == weights.shape[0]), f"select {select.shape}, weights {weights.shape}"
    return jnp.matmul(select, weights)

def do_leaf(context: Dict[str, Any], ss: List[np.ndarray]):
    s = jnp.concatenate(ss, axis=0)
    return get_weight(context, s)

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

def build_maps(context: Dict[str, Any], x: np.ndarray):
    buckets_map = np.zeros(
        (x.shape[0], x.shape[1] * context['buckets']), dtype=np.bool_
    )
    context['order_map'] = np.zeros((x.shape[0], x.shape[1]), dtype=np.int8)
    context['split_points'] = list()

    for f in range(x.shape[1]):
        bins, split_point = pd.qcut(x[:, f], context['buckets'], labels=False, duplicates='drop', retbins=True)
        context['order_map'][:, f] = bins
        sum_bin = None
        for b in range(split_point.size - 1):
            bin = np.flatnonzero(bins == b)
            if sum_bin is None:
                sum_bin = bin
            else:
                sum_bin = np.concatenate((sum_bin, bin), axis=None)
            buckets_map[sum_bin, f * context['buckets'] + b] = 1
        context['split_points'].append(list(np.delete(split_point, (0,))))

    length = list()
    for f in range(x.shape[1]):
        length.append(np.count_nonzero(buckets_map[:, (f + 1) * context['buckets']-1]))
    print("length = ", length)
    return buckets_map

def pyu_global_setup(x: np.ndarray, buckets: int):
    context = dict()
    context['buckets'] = buckets
    buckets_map = build_maps(context, x)

    return buckets_map, context

def pyu_tree_setup(context: Dict[str, Any]):
    context['tree'] = XgbModel.XgbTree()
    return context

def tree_finish(context: Dict[str, Any]) -> XgbModel.XgbTree:
    return context['tree']

def do_split(context: Dict[str, Any], split_bucket: int):
    if split_bucket == -1:
        context['tree'].insert_split_node(-1, float("inf"))
        return context
    else:
        feature = int(split_bucket / context['buckets'])
        split_point_idx = split_bucket % context['buckets']
        context['tree'].insert_split_node(
            feature, context['split_points'][feature][split_point_idx]
        )
        ls = (
            (context['order_map'][:, feature] <= split_point_idx).astype(np.int8)
            .reshape(1, context['order_map'].shape[0])
        )
    return ls, context


class SSXgb:
    def __init__(self, spu: SPU) -> None:
        self.spu = spu

    def _update_predict_tree(self, pred: SPU.Object, dataset: List[PYU.Object],
                             tree: List[XgbModel.XgbTree], weight: SPU.Object):
        assert len(tree) == len(dataset)

        weight_selects = list()
        for idx in range(len(dataset)):
            assert tree[idx].device == dataset[idx].device
            weight_selects.append(
                dataset[idx].device(predict_weight_select)(dataset[idx], tree[idx])
            )

        current = self.spu(predict_tree_weight)(weight_selects, weight)
        if pred:
            return self.spu(lambda x, y: x + y)(pred, current)
        else:
            return current

    def train(self, trees: int, depth: int, buckets: int, dataset:
                List[PYU.Object], y: PYU.Object):
        self.trees = trees
        self.depth = depth
        self.buckets = buckets
        self.pyus = [data.device for data in dataset]
        self.samples = dataset[0].shape[0]

        buckets_map = list()
        self.buckets_size = list()
        self.pyus_context = list()
        for r in range(len(self.pyus)):
            # context['order_map']表示dataset[r]的每个样本在哪个bucket里, 其结构为dataset[r].shape
            # context['split_points']表示dataset[r]的每列特征对应的分裂点, 其结构为包含dataset[r].shape[1]个list（长度为buckets）的list
            # m表示由order_map演变而来的布尔列(每个特征的bucket个列，从左到右是包含关系)
            m, context = self.pyus[r](pyu_global_setup)(dataset[r], self.buckets)
            buckets_map.append(m)
            self.buckets_size.append(m.shape[1]) # dataset[r].shape[1] * buckets
            self.pyus_context.append(context)

        self.spu_context = self.spu(spu_global_setup)(buckets_map, y)
        del buckets_map

        # 初始化pred
        pred = self.spu(init_pred, static_argnums=(0, 1))(0, self.samples)
        model = XgbModel()
        while len(model.trees) < self.trees:
            for r in range(len(self.pyus)):
                self.pyus_context[r] = self.pyus[r](pyu_tree_setup)(self.pyus_context[r])
            self.spu_context = self.spu(spu_tree_setup)(self.spu_context, pred) # 计算grad和hess

            tree, weight = self.train_tree()
            model.trees.append(tree)
            model.weights.append(weight)

            if len(model.trees) < self.trees:
                pred = self._update_predict_tree(pred, dataset, tree, weight)

        return model

    def _split_rank(self, split_bucket: int):
        pre_end_pos = 0
        for r in range(len(self.buckets_size)):
            current_end_pod = pre_end_pos + self.buckets_size[r]
            if split_bucket < current_end_pod:
                return r, split_bucket - pre_end_pos
            pre_end_pos += self.buckets_size[r]

        assert False, "should not be here"

    def train_level(self, nodes_s, level):
        last_level = level == (self.depth - 1)
        l_nodes_s = [nodes_s[idx] for idx in range(len(nodes_s)) if idx % 2 == 0]
        spu_split_buckets, self.spu_context = self.spu(
            find_best_split_bucket, static_argnums=(2,)
        )(self.spu_context, l_nodes_s, last_level)
        split_buckets = list(ppd.get(spu_split_buckets))
        assert len(split_buckets) == len(nodes_s)
        lchilds_s = list()
        for s in split_buckets:
            split_rank, split_rank_idx = self._split_rank(s)
            for r in range(len(self.pyus)):
                if r == split_rank:
                    lchild_s, self.pyus_context[r] = self.pyus[r](do_split)(
                        self.pyus_context[r], split_rank_idx
                    )
                    lchilds_s.append(lchild_s)
                else:
                    self.pyus_context[r] = self.pyus[r](do_split)(
                        self.pyus_context[r], -1
                    )

        assert len(lchilds_s) == len(split_buckets)
        childs_s = self.spu(get_child_select)(nodes_s, lchilds_s)
        return childs_s

    def train_tree(self):
        root_s = self.spu(root_select, static_argnums=(0, ))(self.samples)
        nodes_s = (root_s, )
        weight = []
        for level in range(self.depth + 1):
            if level < self.depth:
                nodes_s = self.train_level(nodes_s, level)
            else:
                weight = self.spu(do_leaf)(self.spu_context, nodes_s)

        tree = [
            self.pyus[r](tree_finish)(self.pyus_context[r])
            for r in range(len(self.pyus))
        ]
        return tree, weight

    def predict(self, dataset: List[PYU.Object], model: XgbModel) -> SPU.Object:
        if len(model.trees) == 0:
            return None
        assert len(dataset) == len(model.trees[0])
        pred = None
        for idx in range(len(model.trees)):
            pred = self._update_predict_tree(
                pred, dataset, model.trees[idx], model.weights[idx]
            )

        return self.spu(sigmoid)(pred)

class MockXgb():
    def __init__(self) -> None:
        return

    def _update_predict_tree(self, pred, dataset, tree: List[XgbModel.XgbTree], weight):
        assert len(tree) == len(dataset)

        weight_selects = list()
        for idx in range(len(dataset)):
            weight_selects.append(predict_weight_select(dataset[idx], tree[idx]))

        current = predict_tree_weight(weight_selects, weight)
        if pred is not None:
            return pred + current
        else:
            return current

    def train(self, trees: int, depth: int, buckets: int, dataset, y):
        self.trees = trees
        self.depth = depth
        self.buckets = buckets
        self.samples = dataset[0].shape[0]

        buckets_map = list()
        self.fed_num = len(dataset)
        self.buckets_size = list()
        self.local_contexts = list()

        for r in range(len(dataset)):
            mbools, context = pyu_global_setup(dataset[r], self.buckets)
            buckets_map.append(mbools)
            self.buckets_size.append(mbools.shape[1])
            self.local_contexts.append(context)

        self.fed_context = spu_global_setup(buckets_map, y)
        del buckets_map

        pred = init_pred(0, self.samples)

        model = XgbModel()
        while len(model.trees) < self.trees:
            for r in range(len(dataset)):
                self.local_contexts[r] = pyu_tree_setup(self.local_contexts[r])

            self.fed_context = spu_tree_setup(self.fed_context, pred)

            tree, weight = self.train_tree()
            model.trees.append(tree)
            model.weights.append(weight)


            print(f"in {len(model.trees)}-th tree, leaf weights: {weight}")
            print(list(zip(tree[0].split_features, tree[0].split_values)))
            print(list(zip(tree[1].split_features, tree[1].split_values)))

            if len(model.trees) < self.trees:
                pred = self._update_predict_tree(pred, dataset, tree, weight)

        return model

    def _split_rank(self, split_bucket: int):
        pre_end_pos = 0
        for r in range(len(self.buckets_size)):
            current_end_pos = pre_end_pos + self.buckets_size[r]
            if split_bucket < current_end_pos:
                return r, split_bucket - pre_end_pos
            pre_end_pos += self.buckets_size[r]

        assert False, "should not be here"

    def train_level(self, nodes_s, level):
        last_level = level == (self.depth - 1)
        l_nodes_s = [nodes_s[idx] for idx in range(len(nodes_s)) if idx % 2 == 0]

        fed_split_buckets, self.fed_context = find_best_split_bucket(self.fed_context, l_nodes_s, last_level)
        split_buckets = list(fed_split_buckets)
        assert len(split_buckets) == len(nodes_s)

        lchilds_s = list()
        for s in split_buckets:
            split_rank, split_rank_idx = self._split_rank(s)
            for r in range(self.fed_num):
                if r == split_rank:
                    lchild_s, self.local_contexts[r] = do_split(self.local_contexts[r], split_rank_idx)
                    lchilds_s.append(lchild_s)
                else:
                    self.local_contexts[r] = do_split(self.local_contexts[r], -1)

        assert len(lchilds_s) == len(split_buckets)
        childs_s = get_child_select(nodes_s, lchilds_s)
        return childs_s

    def train_tree(self):
        root_s = root_select(self.samples)
        nodes_s = (root_s,)
        weight = None
        for level in range(self.depth + 1):
            if level < self.depth:
                nodes_s = self.train_level(nodes_s, level)
            else:
                weight = do_leaf(self.fed_context, nodes_s)

        tree = [
            tree_finish(self.local_contexts[r])
            for r in range(self.fed_num)
        ]
        return tree, weight

    def predict(self, dataset, model):
        if len(model.trees) == 0:
            return None
        assert len(dataset) == len(model.trees[0])
        pred = None
        for idx in range(len(model.trees)):
            pred = self._update_predict_tree(pred, dataset, model.trees[idx], model.weights[idx])

        return sigmoid(pred)

def test_mpc_xgb():
    dataset_config = ppd_init()
    x1, x2, y = load_dataset_by_config(dataset_config)
    x2 = x2[:, :x2.shape[1]-5]
    x1, y = ppd.device("P1")(load_feature_r1)(x1, y)
    x2 = ppd.device("P2")(load_feature_r2)(x2)

    dataset = [x1, x2]

    start = time.time()
    ss_xgb = SSXgb(ppd.device("SPU"))
    model = ss_xgb.train(3, 3, 4, dataset, y)
    train_time = time.time() - start
    print(f"train time {train_time}")

    start = time.time()
    yhat = ppd.get(ss_xgb.predict(dataset, model))
    predict_time = time.time() - start
    print(f"predict time {predict_time}")

    score = roc_auc_score(ppd.get(y), yhat)
    print(f"auc {score}")

def test_mock_xgb():
    dataset_config = None
    with open("ds_breast_cancer_basic.json", "r") as f:
        dataset_config = json.load(f)

    x1, x2, y = load_dataset_by_config(dataset_config)
    x2 = x2[:, :x2.shape[1] - 5]

    print("x1: ", x1)
    print("x2: ", x2)

    dataset = [x1, x2]

    mock_xgb = MockXgb()
    model = mock_xgb.train(3, 3, 4, dataset, y)

    yhat = mock_xgb.predict(dataset, model)
    score = roc_auc_score(y, yhat)
    print(f"auc {score}")

def test_build_maps():
    dataset_config = None
    with open("ds_breast_cancer_basic.json", "r") as f:
        dataset_config = json.load(f)

    x1, x2, y = load_dataset_by_config(dataset_config)

    print(x1)
    print(type(x1), " ", x1.shape)
    print(y)
    print(type(y), " ", y.shape)

    res = jnp.concatenate([x1[:6, :], x2[:6, :]], axis=1)
    print("res ", res.shape)

    xx = np.array([1, 2, 4, 5, 7])
    yy = np.array([2, 6, 7, 8, 9, 13])

    sys.exit(0)

    m, context = pyu_global_setup(x1, 4)

    print(f"context: {context}")

    samples = len(x1)
    root_s = jnp.ones((1, samples))

    nodes_s = (root_s,)
    l_nodes_s = [nodes_s[idx] for idx in range(len(nodes_s)) if idx % 2 == 0]

    print(f"l_nodes_s: {l_nodes_s}")


def preprocess_naive(x, y=None):
    n = len(x)
    native_bin_pos = range(n)
    native_bin_value = [v for v in x[:, 0]]

    remote_bin_pos_arr = [[i] for i in range(n)]
    remote_bin_value_arr = [[v] for v in x[:, 0]]

    cols = x.shape[1] - 1
    dataset = np.zeros((n, cols), dtype=float)
    for i in range(n):
        idx = native_bin_pos[i]
        if idx != -1:
            dataset[i] = x[idx][1:].copy()

    label = y

    return native_bin_pos, native_bin_value, remote_bin_pos_arr, remote_bin_value_arr, dataset, label

def gen_data(path):
    dataset_config = None
    with open(path, "r") as f:
        dataset_config = json.load(f)

    x0, x1, y = load_dataset_by_config(dataset_config)
    x1 = x1[:, :x1.shape[1] - 5]

    n = x0.shape[0]
    mask64 = np.uint64(0xffffffffffffffff)

    id0 = np.array(range(n))  # np.random.randint(0, mask64, n, dtype=np.uint64)
    id1 = np.array(range(n))  # np.random.randint(0, mask64, n, dtype=np.uint64)
    for i in range(n):
        id1[i] = id0[i]
        # if np.random.randint(0, 4, dtype=np.uint8) == 0:
        #     id1[i] = id0[i]

    id0 = id0.reshape(n, 1).astype('O')
    id1 = id1.reshape(n, 1).astype('O')

    assert np.array_equal(id0, id1)

    x0 = x0.astype('O')
    x1 = x1.astype('O')

    x0 = np.concatenate([id0, x0], axis=1)
    x1 = np.concatenate([id1, x1], axis=1)

    return x0, x1, y

def test_xgb_for():
    x1, x2, y = gen_data("ds_breast_cancer_basic.json")
    _, _, _, _, x1, y = preprocess_naive(x1, y)
    _, _, _, _, x2, _ = preprocess_naive(x2)

    shuffle = np.arange(x1.shape[0])
    np.random.shuffle(shuffle)

    x1_shuffled = np.zeros((x1.shape[0], x1.shape[1]), dtype='O')
    x2_shuffled = np.zeros((x2.shape[0], x2.shape[1]), dtype='O')
    y_shuffled = np.zeros(y.shape)

    for i in range(x1.shape[0]):
        idx = shuffle[i]
        x1_shuffled[idx, :] = x1[i, :]
        x2_shuffled[idx, :] = x2[i, :]
        y_shuffled[idx]  = y[i]

    x1 = x1_shuffled.astype(np.float64)
    x2 = x2_shuffled.astype(np.float64)
    y  = y_shuffled

    dataset = [x1, x2]

    mock_xgb = MockXgb()
    model = mock_xgb.train(3, 3, 4, dataset, y)

    yhat = mock_xgb.predict(dataset, model)
    score = roc_auc_score(y, yhat)
    print(f"auc {score}")

if __name__ == "__main__":
    test_mpc_xgb()
    # test_build_maps()
    # test_mock_xgb()
    # test_xgb_for()


    # in 1 - th tree, leaf weights: [0.5788733 - 0.          0.39473686 - 0.33913043  0.20000002 - 0.3
    #           - 0.5914894 - 0.]
    # [(-1, inf), (-1, inf), (4, 1.5260930888575458e-05), (6, 1.7395392571697225e-05), (-1, inf), (0, 0.0038011283497884347), (0, 0.00923366243535496)]
    # [(4, 0.004417019275975553), (4, 0.003519040902679831), (-1, inf), (-1, inf), (8, 3.086506817113305e-05), (-1, inf), (-1, inf)]
    # in 2 - th tree, leaf weights: [0.3619731   0.00297607  0.16286477 - 0.37114453  0.3 - 0.
    #           - 0.04816862 - 0.39948067]
    # [(6, 1.7395392571697225e-05), (-1, inf), (-1, inf), (-1, inf), (0, 0.0038011283497884347), (0, 0.004428772919605078), (3, 2.030324400564175e-05)]
    # [(-1, inf), (4, 0.004417019275975553), (4, 0.003058298072402445), (5, 0.00698636577338975), (-1, inf), (-1, inf), (-1, inf)]
    # in 3 - th tree, leaf weights: [0.34669438 - 0.09921078  0.24631207 - 0.23114489  0.38761413 - 0.07984411
    #           0.15804917 - 0.3539291]
    # [(-1, inf), (5, 3.0724024447578753e-05), (5, 1.4466384579219558e-05), (12, 0.010622943112364832), (-1, inf), (0, 0.004428772919605078), (-1, inf)]
    # [(7, 0.16137752703338035), (-1, inf), (-1, inf), (-1, inf), (4, 0.003058298072402445), (-1, inf), (5, 0.004955336154207804)]
