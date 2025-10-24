# Claim
This file reproduces the effectiveness test in Section 6.2. We claim that **AnonGBDT achieves accuracy comparable to plaintext centralized XGBoost training**, and **outperforms** other MPC-based federated GBDT methods.

# How to run
The datasets are located in `./artifact/data/dataaset`. Details about dataset format and usage can be found in the [Dataset](../../README.md#dataset) section in [README.md](../../README.md).

## Scripts and Commands
For each dataset, two scripts are provided:
- `run_{dataset_name}_p0.sh` for part 0 / Alice
- `run_{dataset_name}_p1.sh` for party 1 / Bob.

Open two terminals and run each script separately (one per party). The `{dataset_name}` can be `a9a`, `breast_cancer`, `cod_rna`, `covtype`, `phishing`, `skin_nonskin`. 

We apply k-fold (k=5) validation in the evaluation. Ecah script will:
1. Train k models (k = 5) using `anongbdt_main` command, the usage guide of this command is shown in `AnonGBDT` section in in [README.md](../../README.md).
2. Run the evaluation script (`acc_evaluation.py`) to compute the *average F1 score* across all 5 folds.

The training outputs:
- the time and communication cost on the sceen ([Performance Result](../../README.md#performance-result) in [READM.md](../../README.md))
- the models for both parties ([Model Output Format](../../README.md#model-output-format) in [READM.md](../../README.md)).

For the evaluation command `acc_evaluation.py`, the usage:
```
$ pythont acc_evaluation.py --help
```
Outputs:
```
usage: acc_evaluation.py [-h] [--test_path TEST_PATH] [--model_path MODEL_PATH] [--dataset DATASET]

XGB inference: avg F1 score over all k-fold test

options:
  -h, --help                show this help message and exit
  --test_path TEST_PATH     test file path
  --model_path MODEL_PATH   model path
  --dataset DATASET         dataset name
```
For example:
```
$ python acc_evaluation.py --test_path ../../artifact/data/dataset/ --model_path ../../artifact/output/model/claim4/ --dataset a9a
```

## Remark
1. Core dumps may occasionally occur: If this happens, copy the command from the shell script and run it manually in the terminal to ensure correct execution.
