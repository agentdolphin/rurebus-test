#!/bin/bash

sudo apt update && \
sudo apt upgrade && \
sudo apt install python3-venv && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
pip install razdel pandas scipy transformers scikit-learn && \
mkdir examples && \
python main.py && \
python run_ner.py --data_dir ./examples --output_dir ./output --bert_model bert-base-multilingual-uncased --train_batch_size 8 --eval_batch_size 8 --task_name ner --do_train --do_validation --do_eval && \
python predictions_writer.py
#python run_multi_learning.py --from_tf False --data_dir ./examples --output_dir ./output --mode 1 --model bert-base-multilingual-uncased --learning_rate 1e-5 --do_train --do_validate --do_eval && \
#python predictions_tagrel_writer.py
