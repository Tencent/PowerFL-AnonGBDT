import multiprocessing
import sys
import unittest
from tempfile import TemporaryDirectory

import multiprocess
import random

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
    
    def test_percept_pir(self):
        link_desc = create_link_desc(2)

        path = ["/home/spu/spu/tests/data/alice.csv",
                "/home/spu/spu/tests/data/bob.csv"]

        manager = multiprocessing.Manager()
        res_dict = manager.dict()
        input_arr = []
        for rank in range(2):
            df = pd.read_csv(path[rank])
            if rank == 0:
                input_arr.append(list(df.iloc[:,1].astype(str)))
            else:
                tuples = list(zip(df.iloc[:,1].astype(str), df.iloc[:,-1].astype(str)))
                input_arr.append(tuples)

        def wrap(rank, link_desc, input_arr, res_dict):
            link_ctx = link.create_brpc(link_desc, rank)
            inputs = input_arr[rank]

            role = "guest"
            if rank != 0:
                role = "host"
                bucket_size = 2
                num_thread = 1
                res = libs.percept_pir_server(link_ctx, inputs, bucket_size, num_thread)
            else:
                role = "guest"
                idx = inputs[random.randint(0, len(inputs))]
                res = libs.percept_pir_client(link_ctx, idx)
                res = (idx, res)

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

        (id, res0) = res_dict[0]
        res1 = res_dict[1]

        host_input = {k:v for (k,v) in input_arr[1]}

        if id in host_input:
            print("host answer: ", res1)
            assert res0 == host_input[id] and res1 == "matched"
        else:
            print("guest answer: ", res0)
            print("host answer: ", res1)
            assert res0 == res1 and res0 == ""

    # def test_percept_pir_new(self):
        link_desc = create_link_desc(2)

        path = ["/home/spu/spu/tests/data/alice.csv",
                "/home/spu/spu/tests/data/bob.csv"]
        
        bucket_num = 32
        num_thread = 4
        
        input_arr = []
        for rank in range(2):
            df = pd.read_csv(path[rank])
            if rank == 0:
                input_arr.append(list(df.iloc[:,1].astype(str)))
            else:
                tuples = list(zip(df.iloc[:,1].astype(str), df.iloc[:,-1].astype(str)))
                input_arr.append(tuples)

        manager = multiprocessing.Manager()
        proc_dict = manager.dict()
        for rank in range(2):
            proc_dict[rank] = libs.PerceptPIR()

        return

        def wrap_init(rank, inputs):
            link_ctx = link.create_brpc(link_desc, rank)
            if rank == 0:
                proc_dict[rank] = libs.PerceptPIR(link_ctx, 1)
                proc_dict[rank].init_client(bucket_num)
            else:
                proc_dict[rank] = libs.PerceptPIR(link_ctx, num_thread)
                proc_dict[rank].init_server(inputs[rank], bucket_num)

        jobs = [
            multiprocess.Process(
                target=wrap_init,
                args=(rank, input_arr),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        print("init Done")

        print("bucket num = ", proc_dict[1].bucket_num())
        return

        def wrap_run(rank, proc_dict, inputs, res_dict):

            if rank == 0:
                tmp = inputs[rank]
                idx = tmp[random.randint(0, len(tmp))]
                res = proc_dict[rank].run_client(idx)
                res = (idx, res)
            else:
                res = proc_dict[rank].run_server(inputs[rank], bucket_num)

            res_dict[rank] = res

        res_dict = manager.dict()
        jobs = [
            multiprocess.Process(
                target=wrap_run,
                args=(rank, proc_dict, input_arr, res_dict),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        (id, res0) = res_dict[0]
        res1 = res_dict[1]

        host_input = {k:v for (k,v) in input_arr[1]}

        if id in host_input:
            print("host answer: ", res1)
            assert res0 == host_input[id] and res1 == "matched"
        else:
            print("guest answer: ", res0)
            print("host answer: ", res1)
            assert res0 == res1 and res0 == ""


if __name__ == "__main__":
    unittest.main()

