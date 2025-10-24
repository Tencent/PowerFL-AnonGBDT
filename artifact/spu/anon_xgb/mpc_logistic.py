from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import jax.numpy as jnp
import jax.lax
import numpy as np
from functools import partial

import spu.utils.simulation as spsim
import spu.spu_pb2 as spu_pb2
import spu

def read_breast_cancer_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    scalar = MinMaxScaler(feature_range=(-2, 2))
    print("scalar = ", scalar)
    cols = X.columns
    X = scalar.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)

    print(X.head(4))

    return X, y

def sigmoid_t1(x, limit: bool = True):
    T0 = 1.0 / 2
    T1 = 1.0 / 4
    ret = T0 + x * T1
    if limit:
        return jnp.select([ret < 0, ret > 1], [0, 1], ret)
    else:
        return ret

def sigmoid_sr(x):
    return 0.5 * (x / jnp.sqrt(1 + jnp.power(x, 2))) + 0.5

def sigmoid(x, method='t1'):
    if method == 't1':
        return sigmoid_t1(x)
    else:
        return sigmoid_sr(x)

def compute_dk_func(x, eps=1e-6, method='norm'):
    if method == 'norm':
        return 1 / (jnp.linalg.norm(x) + eps)
    else:
        return jax.lax.rsqrt(jnp.sum(jnp.square(x)) + eps)

class SSLRSGDClassifier:
    def __init__(self,
                 epochs: int,
                 learning_rate: float,
                 batch_size: int,
                 sig_type: str = 't1',
                 eps: float = 1e-6,
                 dk_method: str = 'norm'):
        assert epochs > 0, f"epochs should > 0"
        assert learning_rate > 0, f"learning_rate should > 0"
        assert batch_size > 0, f"batch_size should > 0"
        assert sig_type in ['t1', 'sr'], f"sig_type should be one of ['t1', 'sr']"
        assert eps > 0, f"eps should > 0"
        assert dk_method in ['norm', 'rsqrt'], f"dk_method should be one of ['norm', 'rsqrt']"

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._sig_type = sig_type
        self._eps = eps
        self._dk_method = dk_method
        self._weights = jnp.zeros(())

    def _update_weights(self, x, y, w, total_batch: int, batch_size: int, dk_arr):
        num_feat = x.shape[1]
        assert w.shape[0] == num_feat + 1, "w shape is mismatch to x"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w = w.reshape((w.shape[0], 1))

        compute_dk = False
        if dk_arr is None:
            compute_dk = True
            dk_arr = []

        for idx in range(total_batch):
            begin = idx * batch_size
            end = min((idx + 1) * batch_size, x.shape[0])
            rows = end - begin

            x_slice = jnp.concatenate(
                (x[begin: end, :], jnp.ones((rows, 1))), axis=1
            )
            y_slice = y[begin:end, :]

            pred = jnp.matmul(x_slice, w)
            pred = sigmoid(pred, method=self._sig_type)

            err = pred - y_slice
            grad = jnp.matmul(jnp.transpose(x_slice), err) / rows

            if compute_dk:
                dk = compute_dk_func(grad, self._eps, self._dk_method)
                dk_arr.append(dk)
            else:
                dk = dk_arr[idx]

            step = self._learning_rate * grad * dk
            w = w - step

        if compute_dk:
            dk_arr = jnp.array(dk_arr)

        return w, dk_arr

    def fit(self, x, y):
        assert len(x.shape) == 2, f"expect x to be 2 dimension array, got {x.shape}"
        assert len(y.shape) == 2, f"expect y to be 2 dimension array, got {y.shape}"

        num_sampe = x.shape[0]
        num_feat = x.shape[1]
        batch_size = min(self._batch_size, num_sampe)
        total_batch = (num_sampe + batch_size - 1) // batch_size

        weights = jnp.zeros((num_feat + 1, 1))
        dk_arr = None

        for _ in range(self._epochs):
            weights, dk_arr = self._update_weights(x, y, weights, total_batch, batch_size, dk_arr)

        self._weights = weights
        self.dk_arr = dk_arr

        return

    def predict_proba(self, x):
        num_feat = x.shape[1]
        w = self._weights
        assert w.shape[0] == num_feat + 1, f"w shape is mismatch to x = {x.shape}"
        assert len(w.shape) == 1 or w.shape[1] == 1, "w should be list or 1D array"
        w.reshape((w.shape[0], 1))

        bias = w[-1, 0]
        w = jnp.resize(w, (num_feat, 1))

        pred = jnp.matmul(x, w) + bias
        pred = sigmoid(pred, method=self._sig_type)

        return pred


def cpu_logistic_test(X, y):
    plain_model = SSLRSGDClassifier(epochs=3, learning_rate=0.1, batch_size=8, sig_type='t1', eps=1e-6, dk_method='norm')
    plain_model.fit(X.values, y.values.reshape(-1, 1))

    predict_prob = plain_model.predict_proba(X.values)
    print(f"CPU auc: {roc_auc_score(y.values, predict_prob)}")

def fit_and_predict(x, y, epochs=3, learning_rate=0.1, batch_size=8, sig_type='t1', eps=1e-6, dk_method='norm'):
    model = SSLRSGDClassifier(epochs=epochs, learning_rate=learning_rate,
                              batch_size=batch_size, sig_type=sig_type,
                              eps=eps, dk_method=dk_method)
    model.fit(x, y)
    return model.predict_proba(x)

def sim_logistic_test(X, y):
    Xvalues = X.values
    yvalues = y.values.reshape(-1, 1)

    sim_cheetah = spsim.Simulator.simple(2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM64)
    result = spsim.sim_jax(sim_cheetah, fit_and_predict)(Xvalues, yvalues)
    print(f"Cheetah Sim auc: {roc_auc_score(y, result)}")

    sim_aby = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
    result = spsim.sim_jax(sim_aby, fit_and_predict)(Xvalues, yvalues)
    print(f"ABY 3 Sim auc: {roc_auc_score(y, result)}")

    sim_spdz = spsim.Simulator.simple(2, spu_pb2.ProtocolKind.SEMI2K, spu_pb2.FieldType.FM64)
    result = spsim.sim_jax(sim_spdz, fit_and_predict)(Xvalues, yvalues)
    print(f"SPDZ Sim auc: {roc_auc_score(y, result)}")

    result = spsim.sim_jax(sim_cheetah, partial(fit_and_predict, batch_size=64))(Xvalues, yvalues)
    print(f"Partial fix BatchSize for Cheetah Sim auc: {roc_auc_score(y, result)}")

    result = spsim.sim_jax(sim_cheetah, partial(fit_and_predict, eps=1e-2))(Xvalues, yvalues)
    print(f"Partial fix Epsilon for Cheetah Sim auc: {roc_auc_score(y, result)}")

    result = spsim.sim_jax(sim_cheetah, partial(fit_and_predict, sig_type='sr'))(Xvalues, yvalues)
    print(f"Partial fix sigmoid type for Cheetah Sim auc: {roc_auc_score(y, result)}")

def mpc_logistic_test(X, y):
    def get_dk(x, y, epochs=3, learning_rate=0.1, batch_size=8, sig_type='t1', eps=1e-6, dk_method='norm'):
        model = SSLRSGDClassifier(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
                                  sig_type=sig_type, eps=eps, dk_method=dk_method)
        model.fit(x, y)
        return model.dk_arr

    Xvalues = X.values
    yvalues = y.values.reshape(-1, 1)

    sim128 = spsim.Simulator.simple(2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM128)
    result = spsim.sim_jax(sim128, fit_and_predict)(Xvalues, yvalues)
    print(f"f auc: {roc_auc_score(y, result)}")

    config_che = spu.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH,
        field=spu.FieldType.FM64,
        fxp_fraction_bits=18,
        enable_pphlo_trace=True,
        enable_pphlo_profile=True
    )
    sim_che = spsim.Simulator(2, config_che)

    inputs = np.arange(1000) / 1000
    result1 = spsim.sim_jax(sim_che, partial(compute_dk_func, method='norm'))(inputs)
    result2 = spsim.sim_jax(sim_che, partial(compute_dk_func, method='rsqrt'))(inputs)

if __name__ == "__main__":
    X, y = read_breast_cancer_data()
    # cpu_logistic_test(X, y)
    # sim_logistic_test(X, y)
    mpc_logistic_test(X, y)