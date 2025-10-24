#!/bin/bash
set -e

# train
anongbdt_main -r 1 -i ../../artifact/data/dataset/breast_cancer_Bob_train_0.csv -o ../../artifact/output/model/claim4/breast_cancer_model_Bob_0.txt --tree-num 5 -t 2
sleep 5

anongbdt_main -r 1 -i ../../artifact/data/dataset/breast_cancer_Bob_train_1.csv -o ../../artifact/output/model/claim4/breast_cancer_model_Bob_1.txt --tree-num 5 -t 2
sleep 5

anongbdt_main -r 1 -i ../../artifact/data/dataset/breast_cancer_Bob_train_2.csv -o ../../artifact/output/model/claim4/breast_cancer_model_Bob_2.txt --tree-num 5 -t 2
sleep 5

anongbdt_main -r 1 -i ../../artifact/data/dataset/breast_cancer_Bob_train_3.csv -o ../../artifact/output/model/claim4/breast_cancer_model_Bob_3.txt --tree-num 5 -t 2
sleep 5

anongbdt_main -r 1 -i ../../artifact/data/dataset/breast_cancer_Bob_train_4.csv -o ../../artifact/output/model/claim4/breast_cancer_model_Bob_4.txt --tree-num 5 -t 2
