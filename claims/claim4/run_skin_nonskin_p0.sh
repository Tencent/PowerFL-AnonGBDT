#!/bin/bash
set -e

# train
anongbdt_main -r 0 -i ../../artifact/data/dataset/skin_nonskin_Alice_train_0.csv -o ../../artifact/output/model/claim4/skin_nonskin_model_Alice_0.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/skin_nonskin_Alice_train_1.csv -o ../../artifact/output/model/claim4/skin_nonskin_model_Alice_1.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/skin_nonskin_Alice_train_2.csv -o ../../artifact/output/model/claim4/skin_nonskin_model_Alice_2.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/skin_nonskin_Alice_train_3.csv -o ../../artifact/output/model/claim4/skin_nonskin_model_Alice_3.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 5

anongbdt_main -r 0 -i ../../artifact/data/dataset/skin_nonskin_Alice_train_4.csv -o ../../artifact/output/model/claim4/skin_nonskin_model_Alice_4.txt -H true -P 2 -N 1 --tree-num 10 -t 8
sleep 10

# evaluation
python acc_evaluation.py --test_path ../../artifact/data/dataset/ --model_path ../../artifact/output/model/claim4/ --dataset skin_nonskin
