#!/bin/bash
set -e

#train
anongbdt_main -r 0 -i ../../artifact/data/dataset/covtype_libsvm_binary_Alice_train_0.csv -o ../../artifact/output/model/claim4/covtype_libsvm_binary_model_Alice_0.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/covtype_libsvm_binary_Alice_train_1.csv -o ../../artifact/output/model/claim4/covtype_libsvm_binary_model_Alice_1.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/covtype_libsvm_binary_Alice_train_2.csv -o ../../artifact/output/model/claim4/covtype_libsvm_binary_model_Alice_2.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/covtype_libsvm_binary_Alice_train_3.csv -o ../../artifact/output/model/claim4/covtype_libsvm_binary_model_Alice_3.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/covtype_libsvm_binary_Alice_train_4.csv -o ../../artifact/output/model/claim4/covtype_libsvm_binary_model_Alice_4.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 10

# evaluation
python acc_evaluation.py --test_path ../../artifact/data/dataset/ --model_path ../../artifact/output/model/claim4/ --dataset covtype_libsvm_binary
