import ray
import spu.libspu.link as link
import spu.libpsi.libs as libs

ray.init()

link_desc = create_link_desc(2)

@ray.remote
class Server:
    def __init__(self):
        self.link_ctx = link.create_brpc(link_desc, 1)
        self.idx = ["111", "222", "333"]

    def run(self):
        self.result = libs.circuit_psi("host", self.link_ctx, self.idx)
        return self.result

server = Server.remote()

@ray.remote
class Client:
    def __init__(self, server):
        self.server = server
        self.link_ctx = link.create_brpc(link_desc, 0)
        self.idx = ["111", "123", "345"]

    def run(self):
        self.result = ray.get()
