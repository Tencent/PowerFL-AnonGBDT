# Claim
This file reproduce **the benchmark of cryptographic primitives** in Section 6.2. We claim that the runtime and communication costs of each component in our system (notably sigmoid and FastPackLWEs) outperform those reported for Squirrel [41].

# How to run
Two scripts are provided.
- `run_p0.sh` for part 0/Alice
- `run_p1.sh` for party 1/Bob.
Open two terminals and run each script separately (one per party).

The scripts evaluate the following secure operations: *argmax*, *greater*, *mux*, *BOIS*, *BinMatVec*, *Sigmoid* and *FastPackLWEs*. All settings match those in Table 1 of the paper.

The scripts internally call the `benchmark` command. Usage details can be found in the [Benchmark](../../README.md#benchmark-benchmark) section in [READM.md](../../README.md). It will output the time and communication cost (if it has), such as:
```
================= Evaluation Result ====================
argmax done [1869 milliseconds], [15008902 B]
========================================================
```


# Remark
1. To test under WAN condition, use the `tc` command as described in [Network Environment Setup](../../README.md#network-environment-setup) section in [READM.md](../../README.md)
2. For the local primitive, FastPackLWEs, only one script (`run_p0.sh` or `run_p1.sh`) need to be executed.
