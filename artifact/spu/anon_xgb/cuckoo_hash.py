import hashlib
import math
import sys
import jax.numpy as jnp

import numpy as np
from Cryptodome.Cipher import AES
import struct

mask64 = 0xffffffffffffffff

class CuckooHasher:
    def __init__(self, seed, hash_num=3):
        self.hash_num = hash_num
        key = hashlib.sha256(np.uint64(seed)).digest()[:16]
        self.enc = AES.new(key, AES.MODE_ECB)

        assert (hash_num >= 2), f"hash number should be at least 2"

    def to_bytes(self, x):
        return struct.pack('<Q', x)

    def from_bytes(self, x):
        return np.frombuffer(x, dtype=np.uint64)[0]

    def _hash(self, x):
        ss = self.to_bytes(x)
        zz = bytes([0,0,0,0,0,0,0,0])
        yy = bytes(self.enc.encrypt(ss + zz)[:8])
        y = self.from_bytes(yy)
        return np.bitwise_xor(x, y)

    def _hash2bucket(self, hashes, nbin, j):
        if j > 0:
            values = np.full(len(hashes), j, dtype=np.uint64)
            return [int(self._hash(x) % nbin) for x in np.bitwise_xor(np.array(hashes), values)]
        else:
            return [int(x % nbin) for x in hashes]

    def hash(self, inputs, nbin):
        assert (nbin >= 4), "bin number must be at least 4"
        assert (len(inputs) < nbin), "input length should be less than bin number"

        hashes = [self._hash(x) for x in inputs]
        result = []
        for j in range(self.hash_num):
            result.append(self._hash2bucket(hashes, np.uint64(nbin), j))

        return result

    def hash_simple(self, inputs, nbin):
        assert nbin > 4, "bin number must be at least 4"

        hashes = self.hash(inputs, nbin)

        bucket_arr = [list() for _ in range(nbin)]
        result_arr = [list() for _ in range(nbin)]
        for j in range(len(inputs)):
            for i in range(self.hash_num):
                idx = hashes[i][j]
                bucket_arr[idx].append(j)
                xx = self.to_bytes(inputs[j]) + self.to_bytes(np.uint64(i))
                vv = self.from_bytes(hashlib.sha256(xx).digest()[:8])
                result_arr[idx].append(vv)

        for i in range(nbin):
            if len(result_arr[i]) == 0:
                result_arr[i].append(np.random.randint(0, mask64, dtype=np.uint64))
                bucket_arr[i].append(-1)

        return bucket_arr, result_arr

    def hash_cuckoo(self, inputs, nbin):
        assert nbin > 4, "bin number must be at least 4"
        assert (len(inputs) < nbin), "input length should be less than bin number"

        buckets = [-1] * nbin
        next_hash_index = [0] * nbin

        num_tries = len(inputs)
        hashes = self.hash(inputs, nbin)

        for i in range(len(inputs)):
            current_element = i
            current_hash_index = 0

            tries = 0
            while tries < num_tries:
                index = hashes[current_hash_index][current_element]

                if buckets[index] == -1:
                    buckets[index] = current_element
                    tries = num_tries + 1
                else:
                    (buckets[index], current_element) = (current_element, buckets[index])
                    (next_hash_index[index], current_hash_index) = (current_hash_index, next_hash_index[index])
                    next_hash_index[index] = (next_hash_index[index] + 1) % self.hash_num
                    tries = tries + 1

        result = [(np.uint64(0), 0) for _ in range(nbin)]
        for idx in range(nbin):
            if buckets[idx] != -1:
                i = buckets[idx]
                j = 0
                while idx != hashes[j][i] and j < self.hash_num:
                    j = j + 1

                assert idx == hashes[j][i], f"some error in hash_cuckoo"

                xx = self.to_bytes(inputs[i]) + self.to_bytes(np.uint64(j))
                vv = self.from_bytes(hashlib.sha256(xx).digest()[:8])
                result[idx] = vv
            else:
                result[idx] = np.random.randint(0, mask64, dtype=np.uint64)

        return buckets, result

def test_cuckoo_hash():
    n = 100
    nbin = math.ceil(1.27 * n)

    xarr = np.random.randint(0, mask64, n, dtype=np.uint64)
    yarr = np.random.randint(0, mask64, n, dtype=np.uint64)
    varr = np.random.randint(0, mask64, n, dtype=np.uint64)
    for i in range(n):
        if np.random.randint(0, 4, dtype=np.uint8) == 0:
            yarr[i] = xarr[i]

    print("xarr generation done")

    bools = np.equal(xarr, yarr)

    cuckoohash = CuckooHasher(np.uint64(0))

    bin_arr_pos, bin_arr_value = cuckoohash.hash_simple(xarr, nbin)
    bin_pos, bin_value = cuckoohash.hash_cuckoo(yarr, nbin)

    comp_bools = np.array([False] * nbin)
    comp_values = np.array([np.uint64(0)] * nbin)
    for i in range(nbin):
        vv = bin_value[i]
        arr = bin_arr_value[i]
        arr_pos = bin_arr_pos[i]

        for j in range(len(arr)):
            if arr[j] == vv:
                comp_bools[i] = True
                comp_values[i] = varr[arr_pos[j]]

    real_bools = np.array([False] * nbin)
    real_values = np.array([np.uint64(0)] * nbin)
    for i in range(nbin):
        if bin_pos[i] != -1:
            idx = bin_pos[i]
            real_bools[i] = bools[idx]
            if bools[idx]:
                real_values[i] = varr[idx]

    print(np.array_equal(comp_bools, real_bools))
    print(np.array_equal(comp_values, real_values))

    aa = np.array(bin_value)
    bb = list()
    for vv in bin_arr_value:
        bb.extend(vv)
    bb = np.array(bb)

    tt = jnp.isin(aa, bb)
    print(np.array_equal(real_bools, tt))

    print("simple hash done")


def test_for_repeat():
    n = 1000
    id0 = np.arange(n).astype(np.uint64)
    id1 = np.arange(n).astype(np.uint64)
    cuckoo_hasher = CuckooHasher(1)

    eps = 1.27
    nbin = math.ceil(n * eps)

    bin_pos0, bin_values0 = cuckoo_hasher.hash_cuckoo(id0, nbin)
    bin_pos1, bin_values1 = cuckoo_hasher.hash_cuckoo(id1, nbin)

    print(bin_pos0)
    print(bin_pos1)

    print(np.array_equal(np.array(bin_pos0), np.array(bin_pos1)))


if __name__ == "__main__":
    test_cuckoo_hash()
    #
    # aa = np.random.randint(0, 2, 100, dtype=np.bool_)
    # bb = np.random.randint(0, 50, 100, dtype=np.uint64)
    # res0 = jnp.multiply(aa, bb)
    # res1 = list()
    # for i in range(len(aa)):
    #     if aa[i]:
    #         res1.append(bb[i])
    #     else:
    #         res1.append(np.uint64(0))

    # print(np.array_equal(res0, res1))

    # test_for_repeat()