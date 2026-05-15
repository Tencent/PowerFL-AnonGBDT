# PowerFL-AnonGBDT

[SP 2026] Official code of the paper ["Practical Anonymous Two-Party Gradient Boosting Decision Tree"](paper/AnonGBDT_SP26_paper.pdf).

Keywords: Multi-party Computation, Federated Learning, Private Set Intersection, Homomorphic Encryption

**Table of Contents**
- [PowerFL-AnonGBDT](#powerfl-anongbdt)
  - [Update Log](#update-log)
  - [Description](#description)
  - [Disclaimer](#disclaimer)
  - [Folder Structure](#folder-structure)
  - [Recommend Computer Specs](#recommend-computer-specs)
  - [Quick Start](#quick-start)
    - [Prerequisite](#prerequisite)
    - [Steps](#steps)
  - [Build from Source](#build-from-source)
    - [Prerequisite](#prerequisite-1)
    - [Install Guide](#install-guide)
  - [Network Environment Setup](#network-environment-setup)
  - [Usage Guide](#usage-guide)
    - [Benchmark (`benchmark`)](#benchmark-benchmark)
    - [AnonGBDT](#anongbdt)
      - [Dataset](#dataset)
      - [AnonGBDT Base Version (`basegbdt_main`)](#anongbdt-base-version-basegbdt_main)
      - [AnonGBDT Advanced Version (`anongbdt_main`)](#anongbdt-advanced-version-anongbdt_main)
      - [Model Output Format](#model-output-format)
      - [Performance Result](#performance-result)
  - [Reproducibility Remarks](#reproducibility-remarks)
  - [Troubleshooting](#troubleshooting)
  - [Acknowledgement](#acknowledgement)

## Update Log
- [2025/10/25] PowerFL-AnonGBDT has been accepted by [S&P 2026](https://sp2026.ieee-security.org/)

## Description

Current federated learning leaks the IDs of both datasets, which may violates the regulations in some circumstances. PowerFL-AnonGBDT hiding them via Circuit private-set intersection (circuit-PSI). We propose the Dual-circuit-PSI framework combing with the oblivious indicator synchronization, achieving an efficient and secure GBDT training/inference.

## Disclaimer
Code is purely for research demo and it's not designed for production.

## Folder Structure
- `./artifact`: Contains source code (`./artifact/libspu/cheetah/cpsi/`) and datasets (`./artifact/data/dataset`).
- `./claims`: Contains all evaluations described in Section 6 of the paper. Each claim includes a guide `claim.md`, expected results `expected.md`, and scripts `*.sh`.


## Recommend Computer Specs
We recommand to use 16 cores CPU (that can support AVX-512) with 64GB memory. 

## Quick Start

### Prerequisite
Docker version 20.10.2 or greater is required. Please run the command to check the version:
```
$ docker version
```
Using the docker `secretflow/ubuntu-base-ci:20240710` as the base: `docker pull secretflow/ubuntu-base-ci:20240710`


### Steps
1. Clone the repo:

    ```
    git clone https://github.com/Tencent/PowerFL-AnonGBDT.git
    ```
2. run the following command:
    ```
    $ cd ./artifact
    $ docker build -t anongbdt:latest .
    ```
    check it via:
    ```
    $ docker images
    ```

3. run the docker via
    ```
    $ docker run -d -it --name anongbdt \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --cap-add=NET_ADMIN \
      --network host \
      --privileged=true \
      anongbdt:latest /bin/bash
    ```

4. enter the docker: `docker exec -it anongbdt bash`

5. Let's take claim1 as an example.
    For party 0:
    ```
    $ cd ./claims/claim1
    $ ./run_p0.sh
    ```
    For paty 1:
    ```
    $ cd ./claims/claim1
    $ ./run_p1.sh
    ```


## Build from Source
### Prerequisite
Centos 7 and Ubuntu 22.04 are recommended.
Install the following dependencies
```
gcc>=11.2, cmake>=3.26, ninja, nasm>=2.15, python>=3.9, bazelisk, xxd, lld
```
About the commands used to install the above dependencies, you can follow [Ubuntu docker file](https://github.com/secretflow/devtools/blob/e5be1ff95f23282547f3ba7e6c246fd16c7f90dd/dockerfiles/ubuntu-base-ci.DockerFile).

### Install Guide
Run `./artifact/install.sh`, it compiles the source code in `./artifact` and generates three binaries we used throughout the whole evaluaiotn:
- `basegbdt_main` - AnonGBDT base version.
- `anongbdt_main` - AnonGBDT advanced version.
- `benchmark` - benchmark test.

The script will move these binaries to `/bin` folder.

## Network Environment Setup
We deploy both party 0 (Alice) and party 1 (Bob) on the same machine over LAN.

To emulate the WAN network, we use `tc` command to limit the bandiwidth to 200mbps and round-trip latency/ping-time to 20ms:
```
$ tc qdisc add dev lo root netem delay 10ms rate 200mbit
``` 
Here, `lo` is the network card (which can be checked by `ifconfig`), 10ms represents one-way latency, resulting in ~20 ms RTT.
Ping command can be used to measure the lantency:
```
$ ping 127.0.0.1
```
After WAN test, remove the limitation:
```
$ tc qdisc del dev lo root
```



## Usage Guide
Evaluation guides are in the `claims` folder, here, we iluustarte the usage of the three binaries.

We use the following naming convention throughout:
- party 0 / P0 / Alice / role=0
- party 1 / P1 / Bob / role=1

### Benchmark (`benchmark`)
View options:
```
$ benchmark --help
```
```
Allowed options:
  -h [ --help ]                         produce this message
  -r [ --role ] arg (=0)                Role of the node
  -a [ --alice-address ] arg (=0.0.0.0) IP address of the Alice
  -b [ --bob-address ] arg (=0.0.0.0)   IP address of the Bob
  -x [ --alice-port ] arg (=18000)      Port of the Alice
  -y [ --bob-port ] arg (=19000)        Port of the Bob
  -p [ --cryptographic-primitive ] arg  Cryptographic primitive, including 
                                        argmax, greater, mux, bois, binmatvec, 
                                        sigmoid, fastpacklwes
  -s [ --log2-scale ] arg (=20)         Log2 scale
  -n [ --number-sample ] arg (=100000)  Number of samples
  -m [ --number-feature ] arg (=10)     Number of features
```
Run in two terminals (one per party) with the same primitive, e.g.:
```
(Alice)$ benchmark -r 0 -p argmax -n 100 -m 1
(Bob)$ benchmark -r 1 -p argmax -n 100 -m 1
```
 It will output the time and communication cost at the end, for example: 

```
================= Evaluation Result ====================
argmax done [1869 milliseconds], [15008902 B]
========================================================
```
### AnonGBDT

Both `basegbdt_main` and `anongbdt_main` support:
- Synthetic dataset - specify `-n` (samples) and `-m` (features); leave `-i`(input file) and -o (output model file) empty.
- Real-world dataset - specify -i (input path) and -o (output path); leave -n and -m empty.

#### Dataset
Available in  ``./artifact/data/dataaset``, including `breast-cancer`, `phishing`, `a9a`, `cod-rna`, `skin_noskin`, `covtype.binary`. We apply 5-fold cross-validation and split each dataset into two files (one for Alice and one for Bob). Both training and testing files are provided. Both trainning and testing files can be found in the folder.

**Input Format**

We support the input format the same as the dataset provided in our `./artifact/data/dataaset` folder.

The data for Alice (`./artifact/data/dataaset/breast_cancer_Alice_train_0.csv`):
```
1,2.0,1000025.0,5.0,1.0,1.0
2,2.0,1002945.0,5.0,4.0,4.0
```
where the 1st column is the ID, the second column is the label, and other columns are features.

The data for Bob (`./artifact/data/dataaset/breast_cancer_Bob_train.csv`)::
```
1,1.0,2.0,1.0,3.0,1.0,1.0
2,5.0,7.0,10.0,3.0,2.0,1.0
```
where the 1st column is the ID and other columns are features.

#### AnonGBDT Base Version (`basegbdt_main`)
Show options:
```
$ basegbdt_main --help
```
```
Allowed options:
  -h [ --help ]                         produce this message
  -r [ --role ] arg                     Role of the node
  -a [ --alice-address ] arg (=0.0.0.0) IP address of the Alice
  -b [ --bob-address ] arg (=0.0.0.0)   IP address of the Bob
  -x [ --alice-port ] arg (=18000)      Port of the Alice
  -y [ --bob-port ] arg (=19000)        Port of the Bob
  -i [ --input-csv-file ] arg           Path and name of input file
  -b [ --bin-file ] arg                 Path and name of bin file
  -B [ --max-bin ] arg (=8)             Max bin: 4,8,16
  -n [ --num-sample ] arg (=10000)      Number of samples (for random synthetic
                                        data)
  -m [ --num-feature ] arg (=10)        Number of features (for random synthetic
                                        data)
  -o [ --output-model-file ] arg        Path and name of output model file
  -d [ --depth ] arg (=5)               Depth of a tree
  -T [ --tree-num ] arg (=1)            Number of trees
  -l [ --lambda ] arg (=0.001)          Lambda
  -s [ --log2-scale ] arg (=20)         Log2 scale
  -P [ --positive-label ] arg (=1)      Positive label
  -N [ --negative-label ] arg (=0)      Negative label
  -H [ --have-label ] arg (=0)          Have label?
```

#### AnonGBDT Advanced Version (`anongbdt_main`)
Show options:
```
$ anongbdt_main --help
```
```
Allowed options:
  -h [ --help ]                         produce this message
  -r [ --role ] arg                     Role of the node
  -a [ --alice-address ] arg (=0.0.0.0) IP address of the Alice
  -b [ --bob-address ] arg (=0.0.0.0)   IP address of the Bob
  -x [ --alice-port ] arg (=18000)      Port of the Alice
  -y [ --bob-port ] arg (=19000)        Port of the Bob
  -i [ --input-csv-file ] arg           Path and name of input file
  -b [ --bin-file ] arg                 Path and name of bin file
  -B [ --max-bin ] arg (=8)             Max bin: 4,8,16
  -n [ --num-sample ] arg (=10000)      Number of samples (for random synthetic
                                        data)
  -m [ --num-feature ] arg (=10)        Number of features (for random synthetic
                                        data)
  -o [ --output-model-file ] arg        Path and name of output model file
  -d [ --depth ] arg (=5)               Depth of a tree
  -T [ --tree-num ] arg (=1)            Number of trees
  -l [ --lambda ] arg (=0.001)          Lambda
  -s [ --log2-scale ] arg (=20)         Log2 scale
  -P [ --positive-label ] arg (=1)      Positive label
  -N [ --negative-label ] arg (=0)      Negative label
  -H [ --have-label ] arg (=0)          Have label?
```


#### Model Output Format
Each model file has three parts:
1. Header: number of trees, tree depth, number of features, floating-point precision.
2. Splits - $d-1$ lines (where d is tree depth), one per layer. Each split is formatted as: `split_feature:[bin_lower_bound, bin_upper_bound)`. If the split belongs to the other party, it is written as `*, *`.
3. Leaf weights — arithmetic secret shares of leaf weights.

The following is a sample of Alice's output model:
```
1 5 4 20
3, [2, 3)
1, [8, 9); *, *
*, *; *, *; *, *; *, *
*, *; *, *; *, *; 1, [9, inf); *, *; 1, [5, 6); 1, [6, 8); *, *
14976576948867929910 11110464571987449830 10468593530976860919 811366444679459018 16854217815491416777 12796810005601951917 13030275323670776471 15139879723731048792 11710380077976007914 11510052505531322605 16685515085621133849 8211178186383082019 4855558791740035568 17990529611579021525 17753055343071670476 473230921369104935
```
#### Performance Result
Both `basegbdt_main` and `anongbdt_main` output the time and communication consumptions in the screen, e.g.:
```
================= Evaluation Result ====================
BaseGBDT training takes 278301 ms, com bytes = 28082241556 B
========================================================
```
`anongbdt_main` additional outputs the result of each components, e.g.:

```
========= Evaluation Result of Each Components =========
run histogram takes 56443 ms, costs 71813152 bytes
find best split takes 11455 ms, costs 165616831 bytes
sync split takes 2164 ms, costs 137422752 bytes
 - sync indicator cost 43091376 bytes
 - update indicator cost 94331376 bytes
cal gradients takes 533 ms, costs 10333138 bytes
cal weight takes 532 ms, costs 10333138 bytes
========================================================
================= Evaluation Result ====================
AnonGBDT training takes 86345 ms, com bytes = 489510821 B
========================================================
```

## Reproducibility Remarks
- Effectiveness reproducibility: The accuracy obtained from the artifact may differ slightly (either higher or lower) from the results reported in the paper. This variation primarily arises from the inherent randomness in the training process. Additionally, different binning procedures may lead to variations in accuracy.
- Efficiency reproducibility: efficiency can vary based on the hardware used. For results similar to those reported in the paper, it is recommended to use the same Intel 8255C CPU. 


## Troubleshooting
1. If you encounter permission issues when running the script, try the following steps:
    ```
    $ chmod +x ${script}
    $ sudo bash ${script}
    ```
2. You may encounter the following problem:
    ```
    terminate called after throwing an instance of 'yacl::EnforceNotMet'
      what():  [Enforce fail at external/psi/psi/rr22/okvs/dense_mtx.h:220] col < cols(). 
    Stacktrace:
    #0 psi::rr22::okvs::Paxos<>::GetGapCols()+0x55cdb8adcb6a
    #1 psi::rr22::okvs::Paxos<>::BackfillBinary()+0x55cdb8ae0a03
    #2 psi::rr22::okvs::Paxos<>::Encode()+0x55cdb8ae2424
    #3 spu::mpc::cheetah::PaxosOKVS::encode()+0x55cdb8ad7a39
    #4 spu::mpc::cheetah::VoleCPsi::OpprfShareSendProc()+0x55cdb8acf05f
    #5 spu::mpc::cheetah::VoleCPsi::OpprfShareSender()+0x55cdb8ad0a1a
    #6 spu::mpc::cheetah::gen_tree_parallel()+0x55cdb8aab5ac
    ```
   This error originates from the SPU library and typically occurs when the #input in circuit-PSI is small. To alleviate this issue, try reducing the number of threads in the script (e.g., change -t 2 to -t 1).

## Acknowledgement
Our code is built on top of [SPU (https://github.com/secretflow/spu)](https://github.com/secretflow/spu), thanks to all the contributors!
