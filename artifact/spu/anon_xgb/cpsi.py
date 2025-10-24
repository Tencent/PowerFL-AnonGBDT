from spu.libpsi import libs
from spu.libspu.link import Context
from typing import List

def circuit_psi(
        link: Context, role: str, input_items: List[str]
) -> bytes:
    return libs.circuit_psi(link, role, input_items)



