import os
import logging
import argparse
from tqdm import tqdm, trange
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification
import predict
from utils import init_logger, load_tokenizer, get_labels
import easydict
import re
WEEKDAY = {0:"월요일",1:"화요일",2:"수요일",3:"목요일",4:"금요일",5:"토요일",6:"일요일"}

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", default="tel4.txt", type=str, help="Input file for prediction")
parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

pred_config = parser.parse_args()

def findFirstSecond(arr):
    second = first = -float('inf')
    second_i = first_i = 0
    for i,n in enumerate(arr):
        if n > first:
            second = first
            first = n
            second_i = first_i
            first_i = i
        elif second < n < first:
            second = n
            second_i = i
    return first_i,second_i

# pred_config = easydict.EasyDict({
#     "input_file":"tel6_in.txt",
#     "output_file":"tel6_out.txt",
#     "model_dir":"./model",
#     "batch_size":32,
#     "no_cuda":"store_true"
# })

args = predict.get_args(pred_config)
device = predict.get_device(pred_config)
model = predict.load_model(pred_config, args, device)
label_lst = get_labels(args)

pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
tokenizer = load_tokenizer(args)
lines = predict.read_input_file(pred_config)
dataset = predict.convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)
sampler = SequentialSampler(dataset)
data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)
all_slot_label_mask = None
preds = None

for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
        
first_pred = []
second_pred = []
for i in range(preds.shape[0]):
    first_pred.append([])
    second_pred.append([])
    for j in range(preds.shape[1]):
        first,second = findFirstSecond(preds[i][j])
        first_pred[i].append(first)
        second_pred[i].append(second)
first_pred = np.array(first_pred)
second_pred = np.array(second_pred)

#최대만 찾기

# slot_label_map = {i: label for i, label in enumerate(label_lst)}
# preds_list = [[] for _ in range(first_pred.shape[0])]

# for i in range(first_pred.shape[0]):
#     for j in range(first_pred.shape[1]):
#         if all_slot_label_mask[i, j] != pad_token_label_id:
#             preds_list[i].append(slot_label_map[first_pred[i][j]])

# 두번째만 찾기

# slot_label_map = {i: label for i, label in enumerate(label_lst)}
# preds_list = [[] for _ in range(second_pred.shape[0])]

# for i in range(second_pred.shape[0]):
#     for j in range(second_pred.shape[1]):
#         if all_slot_label_mask[i, j] != pad_token_label_id:
#             preds_list[i].append(slot_label_map[second_pred[i][j]])


#1,2번째 큰 확률 다 반영

slot_label_map = {i: label for i, label in enumerate(label_lst)}
preds_list = [[] for _ in range(first_pred.shape[0])]

for i in range(first_pred.shape[0]):
    for j in range(first_pred.shape[1]):
        if all_slot_label_mask[i, j] != pad_token_label_id:
            if first_pred[i][j] not in [16,17] and second_pred[i][j] in [16,17]:
                preds_list[i].append(slot_label_map[second_pred[i][j]])
            else:
                preds_list[i].append(slot_label_map[first_pred[i][j]])

date = []
time = []
date_time = {}
loc = []
for i,wp in enumerate(zip(lines, preds_list)):
            date_time[i] = []
            for j,(word, p) in enumerate(zip(wp[0], wp[1])):
                #B-I를 같은 리스트에 담아서 연결성 up    ex) 내일 오전 어떠세요? 10시 좋아요
                if p == 'DAT-B':
                    date.append(word)
                    date_time[i].append(word)
                elif p == 'DAT-I':
                    date.append(word)
                    date_time[i].append(word)
                elif p == 'TIM-B':
                    time.append(word)
                    date_time[i].append(word)
                elif p == 'TIM-I':
                    time.append(word)
                    date_time[i].append(word)
                elif p == 'LOC-B':
                    if preds_list[i][j-1] in ['ORG-B','LOC-B'] and loc != []:
                        loc[-1] = loc[-1] + " "+ word
                    else:
                        loc.append(word)
                elif p == 'LOC-I':
                    loc[-1] = loc[-1] + " "+ word
                elif p == 'ORG-B':
                    if preds_list[i][j-1] in ['ORG-B','LOC-B'] and loc != []:
                        loc[-1] = loc[-1] + " "+ word
                    else:
                        loc.append(word)
                elif p == 'ORG-I':
                    loc[-1] = loc[-1] + " "+ word
                elif (p == 'NUM-B'or p == 'NUM-I')and "시" in word:
                    time.append(word)
                    date_time[i].append(word)
                elif "반" in word and preds_list[i][j-1] in ['TIM-B','TIM-I']:
                    time.append(word)
                

now = datetime.now()
year = now.year
month = now.month
day = now.day
date_fix = 0
hour = now.hour
hour_sub = 0   # 없으면 0 오전 1 오후 2
hour_back = 0
hour_flag = 0
minute = 0
minute_sub = ""
weekday = now.weekday()
next_week = 0
next_day = 0
isWeekday= 0

for t in time:
    if "오전" in t:
        hour_sub = 1
    elif "오후" in t:
        hour_sub = 2
    elif "뒤" in t:
        hour_back = re.sub(r'[^0-9]', '', t)
        minute = now.minute
    elif "후에" in t:
        hour_back = re.sub(r'[^0-9]', '', t)
        minute = now.minute
    elif "시" in t:
        if re.search('\d',t):
            hour = re.sub(r'[^0-9]', '', t)  
        elif "한시" in t:
            hour = 1
            hour_flag = 1
        elif "두시" in t :
            hour = 2
            hour_flag = 1
        elif "세시" in t :
            hour = 3
            hour_flag = 1
        elif "네시" in t:
            hour = 4
            hour_flag = 1
        elif "다섯시" in t :
            hour = 5
            hour_flag = 1
        elif "여섯시" in t:
            hour = 6
            hour_flag = 1
        elif "일곱시" in t :
            hour = 7
            hour_flag = 1
        elif "여덜시" in t :
            hour = 8
        elif "아홉시" in t:
            hour = 9
        elif "열시" in t:
            hour = 10
        elif "열한시" in t:
            hour = 11
        elif "열두시" in t:
            hour = 12
    elif "분" in t:
        minute = re.sub(r'[^0-9]', '', t)
    elif "반" in t:
        minute = 30

if hour_sub == 2 and hour_flag == 1:
    hour += 12
if hour_back != 0:
    hour += hour_back

        

for d in date:
    if "다음주" in d or "다음 주" in d:
        next_week = 1
    elif "이번주" in d:
        next_week = 0
    elif "내일" in d and next_day == 0:
        day += 1
        weekday += 1
        next_day = 1
    elif "오늘" in d:
        day = now.day
        weekday = now.weekday()
    elif "월요일" in d:
        promise_week = 0
        isWeekday = 1
    elif "화요일" in d:
        promise_week = 1
        isWeekday = 1
    elif "수요일" in d :
        promise_week = 2
        isWeekday = 1
    elif "목요일" in d: 
        promise_week = 3
        isWeekday = 1
    elif "금요일" in d:
        promise_week = 4
        isWeekday = 1
    elif "토요일" in d:
        promise_week = 5
        isWeekday = 1
    elif "일요일" in d:
        promise_week = 5
        isWeekday = 1
    elif "일" in d and re.search('\d',d):
        day = re.sub(r'[^0-9]', '', d)
        date_fix = 1
    elif "월" in d and re.search('\d',d):
        month = re.sub(r'[^0-9]', '', d)
    elif "년" in d and re.search('\d',d):
        year = re.sub(r'[^0-9]', '', d)
        if year < 2023:
            year = 2023

if isWeekday==1:
    if weekday > promise_week or next_week == 1:
        day = int(now.day) + promise_week - weekday + 7
    else:
        day = int(now.day) + promise_week - weekday
    weekday = promise_week
        
if loc == []:
    location = "미정"
else:
    location = loc[-1]

if day == now.day:
    if int(hour) < now.hour:
        hour = int(hour) + 12
if next_day == 1 or next_week == 0:
    if 0 < int(hour) < 6:
        hour = int(hour) + 12


weekday = weekday % 7

print("약속 장소 : {} , 약속 시간 : {}년 {}월 {}일  {}시 {}분 ({})".format(location,year,month,day,hour,minute,WEEKDAY[weekday]))