import os
import random
import logging
import datetime
import re
import calendar
import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}


def get_test_texts(args):
    texts = []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.split('\t')
            text = text.split()
            texts.append(text)

    return texts

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
    return first_i, second_i

def get_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)

def convert(model, lines,data_loader,device,args, label_lst,pad_token_label_id):
    # making output from loaded model
    all_slot_label_mask = None
    preds = None
    for batch in data_loader:
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
            
    # choose top probability for entity in output
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


    # change probability to entity name
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(first_pred.shape[0])]
    for_loc_list = [[] for _ in range(first_pred.shape[0])]

    for i in range(first_pred.shape[0]):
        for j in range(first_pred.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                if first_pred[i][j] not in [16,17] and second_pred[i][j] in [16,17]:
                    preds_list[i].append(slot_label_map[second_pred[i][j]])
                else:
                    preds_list[i].append(slot_label_map[first_pred[i][j]])
                    
                if first_pred[i][j] not in [8,9,10,11,16,17] and second_pred[i][j] in [8,9,10,11,16,17]:
                    for_loc_list[i].append(slot_label_map[second_pred[i][j]])
                else:
                    for_loc_list[i].append(slot_label_map[first_pred[i][j]])

    # date, time, loc entity classification
    date = []
    time = []
    loc = []
    date_time_loc = {}
    for i,wp in enumerate(zip(lines, preds_list)):
                date_time_loc[i] = []
                for j,(word, p) in enumerate(zip(wp[0], wp[1])):
                    if p == 'DAT-B':
                        date.append(word)
                    elif p == 'DAT-I':
                        if "뒤" in word or "후" in word:
                            date[-1] = date[-1]+word
                        else:
                            date.append(word)
                    elif p == 'TIM-B':
                        time.append(word)
                    elif p == 'TIM-I':
                        if "뒤" in word or "후" in word:
                            time[-1] = time[-1]+word
                        else:
                            time.append(word)
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
                    elif "반" in word and preds_list[i][j-1] in ['TIM-B','TIM-I']:
                        time.append(word)
                    elif "뒤" in word and preds_list[i][j-1] in ['TIM-B','TIM-I']:
                        time[-1] = time[-1]+word
                    elif "후" in word and preds_list[i][j-1] in ['TIM-B','TIM-I']:
                        time[-1] = time[-1]+word
                    else:
                        continue
                    date_time_loc[i].append(word)

    # additional search for loc
    second_loc = []
    if loc == []:
        date_time_loc = {}
        for i,wp in enumerate(zip(lines, for_loc_list)):
                date_time_loc[i] = []
                for j,(word, p) in enumerate(zip(wp[0], wp[1])):
                    if p in ['DAT-B','DAT-I','TIM-B','TIM-I'] or ((p == 'NUM-B'or p == 'NUM-I')and "시" in word) or ("반" in word and preds_list[i][j-1] in ['TIM-B','TIM-I']):
                        date_time_loc[i].append(word)
                        continue
                    elif p == 'LOC-B':
                        if for_loc_list[i][j-1] in ['ORG-B','LOC-B'] and second_loc != []:
                            second_loc[-1] = second_loc[-1] + " "+ word
                        else:
                            second_loc.append(word)
                    elif p == 'LOC-I':
                        second_loc[-1] = second_loc[-1] + " "+ word
                    elif p == 'ORG-B':
                        if for_loc_list[i][j-1] in ['ORG-B','LOC-B'] and loc != []:
                            second_loc[-1] = second_loc[-1] + " "+ word
                        else:
                            second_loc.append(word)
                    elif p == 'ORG-I':
                        second_loc[-1] = second_loc[-1] + " "+ word
                    else:
                        continue
                    date_time_loc[i].append(word)
                    
                    
    # setting date,time as now
    now = datetime.datetime.now()
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

    # time calculate
    for t in time:
        if "오전" in t:
            hour_sub = 1
        elif "오후" in t:
            hour_sub = 2
        elif "뒤" in t:
            hour_back = re.sub(r'[^0-9]', '', t)
        elif "후" in t:
            hour_back = re.sub(r'[^0-9]', '', t)
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

            
    #date calculate
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
        elif "후" in d:
                if re.search('\d',d):
                    day += re.sub(r'[^0-9]', '', d)
                elif "이틀" in d:
                    day += 2
                elif "사흘" in d:
                    day += 3
                elif "나흘" in d:
                    day += 4
                next_day = 1
        elif "일" in d:
            if re.search('\d',d):
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
            isWeekday = 0
        

    weekday = weekday % 7

    #location processing
    max_i = -1
    max = -1
    for i in range(len(date_time_loc)):
        if max <= len(date_time_loc[i]):
            max = len(date_time_loc[i])
            max_i = i

    ignore = ['에서','라는']
    
    location = "미정"
    if loc == []:
        for line in lines:
            for s in line:
                if "에서" in s or "스타벅스" in s:
                    location = s
        

    else:
        for i in range(len(loc),0,-1):
            if "네네" in loc[i-1]:
                continue
            else:
                location = loc[i-1]
                break

    for s in ignore:
        location = re.sub(s,"",location)
    if location[-1] in ["에","가","도"] :   
            location = location[:-1]
   
    #time processing
    if time == []:
        minute = 0
        hour = 0
    hour = int(hour)
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)

    if hour_sub == 2 and hour_flag == 1:
        hour += 12
    if hour_back != 0:
        hour += hour_back

    if day == now.day:
        if hour < now.hour:
            hour = hour + 12
    if next_day == 1 or next_week == 0:
        if 0 < hour < 6:
            hour = hour + 12

    month_last_day = calendar.monthrange(year, month)[1]

    if minute >= 60:
        minute %= 60
        hour += 1
    if hour >=24 :
        hour %= 24
        day+=1
    if day > month_last_day:
        day %= month_last_day
        month+=1
    if month > 12 :
        month %= 12
        year+=1



    d = datetime.datetime(year,month,day,hour,minute)
    
    return location, d