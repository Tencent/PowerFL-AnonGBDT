import json
import random
import unittest
from tempfile import TemporaryDirectory

import multiprocess
from google.protobuf import json_format

import numpy as np
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

        guest_id = "123"
        len = 10000
        host_key_values = [("123", "abc")] * len
        for i in range(len):
            key = str(random.randint(0, 2**64))
            val = "user"+key
            host_key_values[i] = (key, value)

        guest_id = host_key_values[random.randint(0, len)][0]

        bucket_size = 32
        num_thread = 4

        manager = multiprocessing.Manager()
        res_dict = manager.dict()

        def wrap(rank, link_desc):
            link_ctx = link.create_brpc(link_desc, rank)
            if rank == 0:
                res = libs.percept_pir_client(link_ctx, guest_id)
                res_dict[rank] = res
            else:
                res = libs.percept_pir_server(link_ctx, host_key_values, bucket_size, num_thread)
                res_dict[rank] = res

        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(rank, link_desc, configs),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        str0 = res_dict[0]
        str1 = res_dict[1]

        print("guest result: ", str0, " host result: ", str1)


if __name__ == '__main__':
    unittest.main()