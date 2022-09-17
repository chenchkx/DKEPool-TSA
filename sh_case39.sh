#!/bin/bash

DATA="${1-CASE39}"
DEVICE="${2-1}"
BATCH_SIZE="${3-256}"
HD="${4-64}"
FO="${5-0}"
EP="${6-400}"

python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 0
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 1
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 2
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 3
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 4
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 5
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 6
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 7
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 8
python main.py --dataset ${DATA} --device ${DEVICE} --batch_size ${BATCH_SIZE} --hidden_dim ${HD} --fo_type ${FO} --epochs ${EP} --fold_idx 9
