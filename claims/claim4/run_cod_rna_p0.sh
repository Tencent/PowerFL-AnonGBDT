#!/bin/bash
set -e

# train
anongbdt_main -r 0 -i ../../artifact/data/dataset/cod_rna_Alice_train_0.csv  -o ../../artifact/output/model/claim4/cod_rna_model_Alice_0.txt -H true -P 1 -N -1 --tree-num 10 -t 4 
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/cod_rna_Alice_train_1.csv  -o ../../artifact/output/model/claim4/cod_rna_model_Alice_1.txt -H true -P 1 -N -1 --tree-num 10 -t 4 
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/cod_rna_Alice_train_2.csv  -o ../../artifact/output/model/claim4/cod_rna_model_Alice_2.txt -H true -P 1 -N -1 --tree-num 10 -t 4 
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/cod_rna_Alice_train_3.csv  -o ../../artifact/output/model/claim4/cod_rna_model_Alice_3.txt -H true -P 1 -N -1 --tree-num 10 -t 4 
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/cod_rna_Alice_train_4.csv  -o ../../artifact/output/model/claim4/cod_rna_model_Alice_4.txt -H true -P 1 -N -1 --tree-num 10 -t 4 
sleep 10

# evaluation
python acc_evaluation.py --test_path ../../artifact/data/dataset/ --model_path ../../artifact/output/model/claim4/ --dataset cod_rna
