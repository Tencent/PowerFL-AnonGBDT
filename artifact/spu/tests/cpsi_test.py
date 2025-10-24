import multiprocessing
import sys
import unittest
from tempfile import TemporaryDirectory

import multiprocess

import pandas as pd
import spu.libspu.link as link
import spu.libpsi.libs as libs
from .utils import create_link_desc

class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    def test_circuit_psi(self):
        link_desc = create_link_desc(2)

        path = ["./spu/tests/data/alice.csv",
                "./spu/tests/data/bob.csv"]

        manager = multiprocessing.Manager()
        res_dict = manager.dict()
        input_arr = [list(pd.read_csv(path[rank]).iloc[:,1].values) for rank in range(2)]

        def wrap(rank, link_desc, inputs, res_dict):
            link_ctx = link.create_brpc(link_desc, rank)
            idx = inputs[rank]
            role = "guest"
            if rank != 0:
                role = "host"

            res = libs.circuit_psi(role, link_ctx, idx)
            # print(f"{rank} done ", type(idx))
            res_dict[rank] = res

        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(rank, link_desc, input_arr, res_dict),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        (bins, bshr0) = res_dict[0]
        (bin_arr, bshr1) = res_dict[1]

        assert len(bins) == len(bin_arr)
        assert len(bins) == len(bshr0) and len(bins) == len(bshr1)

        for i in range(len(bins)):
            assert len(bins[i]) <= 1
            if len(bins[i]) == 1:
                idx = bins[i][0]
                assert (bshr0[i] ^ bshr1[i]) == int(input_arr[0][idx] in input_arr[1])

if __name__ == "__main__":
    unittest.main()

