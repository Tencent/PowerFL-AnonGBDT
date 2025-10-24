# Claim
This file reproduces the **scalability test** from Section 6.3.

We claim that our system **scales effectively** and remains **compatible with the state-of-the-art MPC GBDT system Squirrel** [41], while relying on **less privacy-compromising** schemes.

# How to run
Open two terminals and run `run.sh` separately (one per party).

Both contains several evaluations, the setting of them are the same as it in Figure 8 (in the paper). To run a specific evaluation, use the following command:
```
$ ./run.sh -r ${role} -s ${setting}
```
Party 0 should runs with `role=0` and party 1 should run with `role=1`.

To view an explanation of the remaining parameters, run:
```
$ ./run.sh -h
```
It outpus
```
run the whole protocol
  ./run.sh -r [role] -s [setting]
role: 0 for party 0, 1 for party 1
setting 1: n = 10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 2: n = 2.5*10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 3: n = 5*10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 4: n = 10^6, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 5: n = 10^5, D = 3, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 6: n = 10^5, D = 4, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 7: n = 10^5, D = 6, B = 16, m0 = 10, m1 = 10, #threads = 8
setting 8: n = 10^5, D = 5, B = 16, m0 = 25, m1 = 25, #threads = 8
setting 9: n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50, #threads = 8
setting 10: n = 10^5, D = 5, B = 16, m0 = 100, m1 = 100, #threads = 8
```
*Important*: Ensure both parties run with the same protocol and setting.


The scripts internally call the `anongbdt_main` command. Usage details can be found in the [AnonGBDT](../../README.md#anongbdt) in [READM.md](../../README.md). It will output the time and communication cost on the sceen ([Performance Result](../../README.md#performance-result) in [READM.md](../../README.md)).


# Remark
1. To test under WAN condition, use the `tc` command as described in [Network Environment Setup](../../README.md#network-environment-setup) section in [READM.md](../../README.md)
2. In the paper, we train 10 trees. However, this is very time-consuming, especially under the WAN setting. Therefore, the provided scripts are configured to train only 1 tree.
