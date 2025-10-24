# Claim
This file reproducse the **efficiency test** in Section 6.2. We claim that our system is compatible with the state-of-the-art MPC federated GBDT training systems[19,41,56], while using less privacy-compromising schemes.

# How to run
Open two terminals and run `run.sh` separately (one per party).


Each script contains four evaluations with the same settings as Table 4 in the paper.
To run a specific evaluation with `${setting}` (1, 2, 3 or 4), use the following command for each:
```
$ ./run.sh -r ${role} -s ${setting}
```
Party 0 should runs with `role=0` and party 1 should run with `role=1`.

To check the specific setting, run:
```
$ ./run.sh -h
```
It will output:
```
run the whole protocol
  ./run.sh -r [role] -s [setting]
role: 0 for party 0, 1 for party 1
setting 1: n = 5*10^4, D = 4, B = 8, m0 = 8, m1 = 7, #threads = 6, LAN
setting 2: n = 2*10^5, D = 4, B = 8, m0 = 8, m1 = 7, #threads = 6, LAN
setting 3: n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16, #threads = 8, LAN
setting 4: n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16, #threads = 8, WAN 100mbps (Please manually configure the traffic control via tc command)
```

*Important*: Ensure both parties run with the same setting.

The scripts internally call the `anongbdt_main` command. Usage details can be found in the [AnonGBDT](../../README.md#anongbdt) in [READM.md](../../README.md). It will output the time and communication cost on the sceen ([Performance Result](../../README.md#performance-result) in [READM.md](../../README.md)).

# Remark: 
1. To test under WAN condition, use the `tc` command as described in [Network Environment Setup](../../README.md#network-environment-setup) section in [READM.md](../../README.md)
