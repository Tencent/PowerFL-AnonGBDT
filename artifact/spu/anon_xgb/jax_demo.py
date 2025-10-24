import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import spu.spu_pb2
import spu.utils.distributed as ppd
from sklearn import metrics
import spu.spu_pb2 as psi

import cuckoofilter


def make_rand():
    np.random.seed()
    return np.random.randint(100, size=(1,))

def greater(x, y):
    return jnp.greater(x, y)

def init_ppd(config=None):
    if config is None:
        ppd.init(ppd.SAMPLE_NODES_DEF, ppd.SAMPLE_DEVICES_DEF)
    else:
        with open(config, 'r') as file:
            conf = json.load(file)
        ppd.init(conf['nodes'], conf['devices'])

    ppd_current = ppd.current()
    print(ppd_current.nodes_def)
    print(ppd_current.devices)

def test():
    x = make_rand()
    y = make_rand()
    ans = greater(x, y)

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x > y = {ans}")

def test_millionares():
    init_ppd()

    x = ppd.device("P1")(make_rand)()
    y = ppd.device("P2")(make_rand)()
    ans = ppd.device("SPU")(greater)(x, y)

    print(x, " ", y, " ", ans)

    print("x > y = ", ppd.get(ans))
    x_revealed = ppd.get(x)
    y_revealed = ppd.get(y)
    print("real: ", x_revealed, " ", y_revealed, " ", np.greater(x_revealed, y_revealed))

def test_logistic_regression():
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    def predict(x, w, b):
        return sigmoid(jnp.matmul(x, w) + b)

    def loss(x, y, w, b):
        pred = predict(x, w, b)
        label_prob = pred * y + (1 - pred) * (1 - y)
        return -jnp.mean(jnp.log(label_prob))

    def train(feature, label, n_epochs=10, n_iters=10, step_size=0.1):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        xs = jnp.array_split(feature, n_iters, axis=0)
        ys = jnp.array_split(label, n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for (x, y) in zip(xs, ys):
                grad = jax.grad(loss, argnums=(2, 3))(x, y, w_, b_)
                w_ -= grad[0] * step_size
                b_ -= grad[1] * step_size
            return w_, b_

        return jax.lax.fori_loop(0, n_epochs, body_fun, (w, b))

    def load_dataset():
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer()

        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        return normalize(ds['data']), ds['target'].astype(dtype=np.float64)

    x, y = load_dataset()
    print("x length = ", x.shape)
    print("y size = ", y.shape)
    w, b = jax.jit(train)(x, y)

    print("CPU AUC = ", metrics.roc_auc_score(y, predict(x, w, b)))

    ## mpc_logistic_regression
    init_ppd()
    X, _ = ppd.device("P1")(load_dataset)()
    _, Y = ppd.device("P2")(load_dataset)()

    W, B = ppd.device("SPU")(train)(X, Y)

    w_ = ppd.get(W)
    b_ = ppd.get(B)

    print("SPU AUC = ", metrics.roc_auc_score(y, predict(x, w_, b_)))

    print("CPU result: ", w)
    print("SPU result: ", w)

def test_set_in():
    max_num = 0xffffffff
    size = 1500000
    x = np.random.randint(0, max_num, size)
    y = np.random.randint(0, max_num, size)

    def load_value(x):
        return x

    def array_equal(x, y):
        return jnp.equal(x, y)

    init_ppd("3pc.json")

    x = ppd.device("P1")(load_value)(x)
    y = ppd.device("P2")(load_value)(y)
    ans = ppd.device("SPU")(array_equal)(x, y)

    spu.spu_pb2.ValueMeta

    print("x equals y: ", ppd.get(ans))
    # x_revealed = ppd.get(x)
    # y_revealed = ppd.get(y)
    # print("real: ", x_revealed, " ", y_revealed, " ", array_equal(x_revealed, y_revealed))

    # def is_set_in(x, y):
    #     if


def test():
    init_ppd()

    @ppd.device("SPU")
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    print(sigmoid)
    print(np.random.rand(3, 3))
    print(sigmoid.dump_pphlo(np.random.rand(3, 3)))

def test_cuckoo():
    cf = cuckoofilter.CuckooFilter(capacity=100, fingerprint_size=1)

if __name__ == "__main__":
    # test_ppd()
    # test_millionares()
    # test_logistic_regression()
    test_set_in()