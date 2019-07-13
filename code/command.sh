#!/usr/bin/env bash

#1.train
python train.py --lr_train_path ../data/lr_train --hr_train_path ../data/hr_train --lr_eval_path ../data/lr_eval --hr_eval_path ../data/hr_eval --patch_size 64 --train_batch_size 16 --eval_batch_size 1 --lr 1e-4 --num_epochs 200 --gpus 2

#2.test
python test.py --weights_file ./saved_weights/rdn_x4.pth --test_path ../data/lr_test --gpus 1
