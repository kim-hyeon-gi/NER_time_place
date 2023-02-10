import os
import logging
import argparse
from tqdm import tqdm, trange
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification
import predict
from utils import init_logger, get_labels, findFirstSecond, MODEL_CLASSES, convert
import easydict
import re
import calendar
WEEKDAY = {0:"월요일",1:"화요일",2:"수요일",3:"목요일",4:"금요일",5:"토요일",6:"일요일"}


# setting
for i in range(11):
    s = "datahub/input"+str(i+1)+".txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=s, type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--tokenizer_dir", default="./tokenizer",type=str, help = "path to save, load tokenizer")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    pred_config = parser.parse_args()

    # trained setting load
    args = predict.get_args(pred_config)
    device = predict.get_device(pred_config)
    model = predict.load_model(pred_config, args, device)
    label_lst = get_labels(args)
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained('./tokenizer')

    # input data processing
    lines = predict.read_input_file(pred_config)
    dataset = predict.convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    # making output from loaded model
    location , d = convert(model, lines,data_loader,device,args, label_lst,pad_token_label_id)
    print("input file : input{}.txt   약속 장소 : {} , 약속 시간 : {}".format(i+1,location,d))