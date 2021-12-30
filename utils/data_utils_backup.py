import json
from datetime import datetime, timedelta
import torch
import numpy as np
import sklearn.metrics as met
import sklearn.multiclass
import math
import random
from transformers import BertTokenizer
import utils.zero_shot_utils as zs

severity_list = ['REAL', 'NOT_SEVERE', 'POSSIBLY_SEVERE', 'HIGHLY_SEVERE']
stance_list = ['SUPPORT', 'DENY', 'NEITHER']
rebuttal_list = ['TRUE', 'FALSE']

def create_tweet_dict(tweet_json_file):
    tweet_dict = {}
    data = []
    with open(tweet_json_file, 'r+') as file:
        for line in file:
            #print(json.loads(line))
            data.append(json.loads(line))
            #break
    for item in data:
        #print(item)
        id = int(item['id'])
        created_at = item['created_at']
        full_text = item['full_text']
        in_reply_to = int(item['in_reply_to_status_id']) if item['in_reply_to_status_id'] != None else None
        user_id = item['user']['id']
        user_name = item['user']['name']
        user_screen_name = item['user']['screen_name']
        user_loc = item['user']['location']
        user_description = item['user']['description']
        user_follower_count = item['user']['followers_count']
        user_friend_count = item['user']['friends_count']
        user_created = item['user']['created_at']
        user_fav_count = item['user']['favourites_count']
        user_verif = item['user']['verified']
        user_statuses_count = item['user']['statuses_count']

        tweet_dict[id] = {}
        new_created_at = datetime.strftime(datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'),
                                         '%Y-%m-%d %H:%M:%S')
        tweet_dict[id]['created_at'] = new_created_at
        #new_datetime = datetime.strftime(datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        tweet_dict[id]['full_text'] = full_text
        tweet_dict[id]['in_reply_to'] = in_reply_to

        tweet_dict[id]['user_id'] = user_id
        tweet_dict[id]['user_name'] = user_name
        tweet_dict[id]['user_screen_name'] = user_screen_name
        tweet_dict[id]['user_loc'] = user_loc
        tweet_dict[id]['user_description'] = user_description
        tweet_dict[id]['user_follower_count'] = user_follower_count
        tweet_dict[id]['user_friend_count'] = user_friend_count
        new_user_created_at = datetime.strftime(datetime.strptime(user_created, '%a %b %d %H:%M:%S +0000 %Y'),
                          '%Y-%m-%d %H:%M:%S')
        tweet_dict[id]['user_created'] = new_user_created_at
        tweet_dict[id]['user_fav_count'] = user_fav_count
        tweet_dict[id]['user_verif'] = user_verif
        tweet_dict[id]['user_statuses_count'] = user_statuses_count

    print(f'Length of tweet_dict: {len(tweet_dict)}')
    return tweet_dict

def load_tweet_dict(tweet_file):
    tweet_dict = {}
    with open(tweet_file, 'r', encoding='utf-8') as file:
        tweet_dict = json.load(file)

    print(f'Loaded {len(tweet_dict)} datapoints.')
    ### MAKE SURE IT'S AN INT
    tweet_dict = {int(k):v for k,v in tweet_dict.items()}
    return tweet_dict

def create_twitter_graph(tweet_dict):
    tweet_graph = set()
    for id in tweet_dict.keys():
        # in reply to id
        ir2_id = tweet_dict[id]['in_reply_to']
        # if this tweet is in response to another tweet, create an edge between the two ids
        if ir2_id in tweet_dict.keys():
            edge = (id, ir2_id)
        else: #otherwise, there is no edge
            edge = (id, None)

        tweet_graph.add(edge)
    return tweet_graph

def print_rebuttal_tweets(tweet_dict, anno_dict):
    count = 0
    for id in anno_dict.keys():
        #print(anno_dict[id]['rebuttal'])
        if anno_dict[id]['rebuttal'] == 0:
            if id in tweet_dict.keys():
                print('\n' + '='*30)
                print(f'{count+1} - {severity_list[anno_dict[id]["severity"]]}: \n{tweet_dict[id]["full_text"]}')
                count += 1
    print('='*20 + ' END ' + '='*30)

def return_x_each(tweet_dict, anno_dict, x):
    count = 0

    for idx, label in enumerate(severity_list):
        print('\n' + '=' * 70)
        print('=' * 20 + f' START {label} ' + '=' * 30)
        print('=' * 70)
        for id in anno_dict.keys():
            #print(anno_dict[id])
            if int(anno_dict[id]['severity']) == idx and count < x:
                if id in tweet_dict.keys():
                    print('\n' + '=' * 40)
                    print(f"{count + 1} - {label}: \n{tweet_dict[id]['full_text']}")
                    count += 1
            if count >= x:
                break
        count = 0
        print('=' * 70)
        print('=' * 20 + f' END {label} ' + '=' * 30)
        print('\n' + '=' * 70)


def create_tweet_label_dict(tweet_anno_file):
    # annotation file will be in the form of:
    # {annoid_1: {severity:'severity', stance:'stance', rebuttal:'rebuttal')}
    # Severity embedding
    se_emb = {}
    for idx, severity in enumerate(severity_list):
        se_emb[severity] = idx
    # Stance embedding
    st_emb = {}
    for idx, stance in enumerate(stance_list):
        st_emb[stance] = idx

    rebut_emb = {}
    for idx, rebuttal in enumerate(rebuttal_list):
        rebut_emb[rebuttal] = idx

    anno_dict = {}

    with open(tweet_anno_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for id in data.keys():
        int_id = int(id)
        if data[id][0] in se_emb.keys():
            anno_dict[int_id] = {}

            # Severity (int), Stance (int), Rebuttal (bool)
            anno_dict[int_id]['severity'] = se_emb[data[id][0]]
            anno_dict[int_id]['stance'] = st_emb[data[id][1]]
            anno_dict[int_id]['rebuttal'] = rebut_emb[data[id][2]]
            anno_dict[int_id]['topic'] = data[id][3]

    print(f'Length of annotation dict: {len(anno_dict)}')
    return anno_dict

def count_stats_labels(anno_dict):
    count_dict = {}
    for id in anno_dict.keys():
        for s in anno_dict[id].keys():
            if s not in count_dict.keys():
                count_dict[s] = {}

            thing = anno_dict[id][s]
            if thing not in count_dict[s].keys():
                count_dict[s][thing] = 0
            else:
                count_dict[s][thing] += 1


    print(f'===========================================')
    for key in count_dict.keys():
        print(f'{key}')
        for thing in count_dict[key].keys():
            print(f'{thing}: {count_dict[key][thing]}')
    print(f'===========================================')

def retrieve_stance_from_emb(stance):
    return stance_list[stance]

def retrieve_severity_from_emb(severity):
    return severity_list[severity]

def retrieve_anno_ids(anno_dict, tweet_dict, triplets=None):

    #sentences = []
    input_ids = []
    attention_masks = []
    user_infos = []
    sev_labs = []
    st_labs = []
    rebut_labs = []
    topic_labs = []

    #print(tweet_dict['1231602135139258400'])
    location_emb = {}
    if triplets == None:
        for id in anno_dict:
            #print(type(id))
            if id in tweet_dict.keys():
                ### GRAB USER INFORMATION FIRST ###
                user_info = []
                # For each user, we want 'follower_count', 'friend_count', 'created acct time - created tweet time',
                # 'favorite count', 'verified', 'status_count', 'location'
                # ex [fol_c, fr_c, fav_c, st_c, cr, ver, loc]
                # location must be embedded if applicable
                if tweet_dict[id]['user_loc'] not in location_emb.keys():
                    location_emb[tweet_dict[id]['user_loc']] = len(location_emb)
                fol_c = int(tweet_dict[id]['user_follower_count'])
                fr_c = int(tweet_dict[id]['user_friend_count'])
                fav_c = int(tweet_dict[id]['user_fav_count'])
                st_c = int(tweet_dict[id]['user_statuses_count'])

                #print(datetime.strptime(tweet_dict[id]['created_at'], '%Y-%m-%d %H:%M:%S'))
                time_diff = datetime.strptime(tweet_dict[id]['created_at'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(tweet_dict[id]['user_created'], '%Y-%m-%d %H:%M:%S')

                cr = int(time_diff.total_seconds())
                ver = int(tweet_dict[id]['user_verif'])
                loc = location_emb[tweet_dict[id]['user_loc']]

                user_info= [fol_c, fr_c, fav_c, st_c, cr, ver, loc]

                ### GRAB SENTENCE ###
                sentence = tweet_dict[id]['full_text']

                ### GRAB SEVERITY ###
                sev_lab = anno_dict[id]['severity']

                ### GRAB STANCE ###
                st_lab = anno_dict[id]['stance']

                ### GRAB REBUTTAL ###
                rebut_lab = anno_dict[id]['rebuttal']

                ### GRAB TOPIC ###
                topic_lab = anno_dict[id]['topic']

                #print(user_info, sev_lab, st_lab, rebut_lab)
                #break
                ### PUT TOGETHER ###
                sentences.append(sentence)
                user_infos.append(user_info)
                sev_labs.append(sev_lab)
                st_labs.append(st_lab)
                rebut_labs.append(rebut_lab)
                topic_labs.append(topic_lab)

        #print(f'Length of sentences is {len(sentences)}')
        #print(f'Length of user_infos is {len(user_infos)}')

        #print(f'Length of user_info[0] is {len(user_infos[0])} and it is {user_infos[0]}')
        # return sentences, user_infos, sev_labs, st_labs, rebut_labs

    else:
        for triplet in triplets:
            anchor = triplet[0]
            positive = triplet[1]
            negative = triplet[2]
            if anchor in tweet_dict.keys() and anchor in anno_dict.keys() and \
                positive in tweet_dict.keys() and positive in anno_dict.keys() and \
                negative in tweet_dict.keys() and negative in anno_dict.keys():
                ### GRAB USER INFORMATION FIRST ###
                user_info = []
                # For each user, we want 'follower_count', 'friend_count', 'created acct time - created tweet time',
                # 'favorite count', 'verified', 'status_count', 'location'
                # ex [fol_c, fr_c, fav_c, st_c, cr, ver, loc]
                # location must be embedded if applicable
                if tweet_dict[anchor]['user_loc'] not in location_emb.keys():
                    location_emb[tweet_dict[anchor]['user_loc']] = len(location_emb)
                fol_c = int(tweet_dict[anchor]['user_follower_count'])
                fr_c = int(tweet_dict[anchor]['user_friend_count'])
                fav_c = int(tweet_dict[anchor]['user_fav_count'])
                st_c = int(tweet_dict[anchor]['user_statuses_count'])

                # print(datetime.strptime(tweet_dict[id]['created_at'], '%Y-%m-%d %H:%M:%S'))
                time_diff = datetime.strptime(tweet_dict[anchor]['created_at'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(
                    tweet_dict[anchor]['user_created'], '%Y-%m-%d %H:%M:%S')

                cr = int(time_diff.total_seconds())
                ver = int(tweet_dict[anchor]['user_verif'])
                loc = location_emb[tweet_dict[anchor]['user_loc']]

                user_info = [fol_c, fr_c, fav_c, st_c, cr, ver, loc]

                ### GRAB SENTENCE ###
                # sentence = tweet_dict[anchor]['full_text']
                # pos_sentence = tweet_dict[positive]['full_text']
                # neg_sentence = tweet_dict[negative]['full_text']
                anchor_input_ids = anno_dict[anchor]['input_ids']
                positive_input_ids = anno_dict[positive]['input_ids']
                negative_input_ids = anno_dict[negative]['input_ids']

                anchor_mask = anno_dict[anchor]['attention_mask']
                positive_mask = anno_dict[positive]['attention_mask']
                negative_mask = anno_dict[negative]['attention_mask']

                triplet_input_ids = (anchor_input_ids, positive_input_ids, negative_input_ids)
                triplet_attention_masks = (anchor_mask, positive_mask, negative_mask)

                ### GRAB SEVERITY ###
                sev_lab = anno_dict[anchor]['severity']

                ### GRAB STANCE ###
                st_lab = anno_dict[anchor]['stance']

                ### GRAB REBUTTAL ###
                rebut_lab = anno_dict[anchor]['rebuttal']

                ### GRAB TOPIC ###
                topic_lab = anno_dict[anchor]['topic']

                # print(user_info, sev_lab, st_lab, rebut_lab)
                # break
                ### PUT TOGETHER ###
                input_ids.append(triplet_input_ids)
                attention_masks.append(triplet_attention_masks)
                # sentences.append(sentence_triplet)
                user_infos.append(user_info)
                sev_labs.append(sev_lab)
                st_labs.append(st_lab)
                rebut_labs.append(rebut_lab)
                topic_labs.append(topic_lab)

    return input_ids, attention_masks, user_infos, sev_labs, st_labs, rebut_labs, topic_labs

def retrieve_ids_and_tensify(sentences, user_infos, sev_labs, st_labs, rebut_labs, topic_labs, tokenizer, maximum_length):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # PYTORCH TENSORS
            truncation=True  ### ADDED MYSELF TO REMOVE WARNING
        )
        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])

        # and its attention mask
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)

    attention_masks = torch.cat(attention_masks, dim=0)

    user_infos = torch.tensor(user_infos, dtype=torch.int64)
    sev_labs = torch.tensor(sev_labs, dtype = torch.int64)
    st_labs = torch.tensor(st_labs, dtype=torch.int64)
    rebut_labs = torch.tensor(rebut_labs, dtype=torch.int64)
    topic_labs = torch.tensor(topic_labs, dtype=torch.int64)

    return input_ids, attention_masks, user_infos, sev_labs, st_labs, rebut_labs, topic_labs

def encode_sentences(anno_dict, tweet_dict, tokenizer, maximum_length):
    for id in anno_dict.keys():
        if id in tweet_dict.keys():
            encoded_dict = tokenizer.encode_plus(
                tweet_dict[id]['full_text'],
                add_special_tokens=True,
                max_length=maximum_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',  # PYTORCH TENSORS
                truncation=True  ### ADDED MYSELF TO REMOVE WARNING
            )
            anno_dict[id]['input_ids'] = encoded_dict['input_ids']
            anno_dict[id]['attention_mask'] = encoded_dict['attention_mask']

    return anno_dict
def ret_ids_from_triplets(input_ids, attention_masks, user_infos, sev_labs, st_labs, rebut_labs, topic_labs, tokenizer, maximum_length):
    anchor_input_ids = []
    anchor_attention_masks = []

    positive_input_ids = []
    positive_attention_masks = []

    negative_input_ids = []
    negative_attention_masks = []

    for iids, attn_m in zip(input_ids, attention_masks):
        anchor_input_ids.append(iids[0])
        anchor_attention_masks.append(attn_m[0])

        positive_input_ids.append(iids[1])
        positive_attention_masks.append(attn_m[1])

        negative_input_ids.append(iids[2])
        negative_attention_masks.append(attn_m[2])


    # Convert the lists into tensors
    anchor_input_ids = torch.cat(anchor_input_ids, dim=0)
    positive_input_ids = torch.cat(positive_input_ids, dim=0)
    negative_input_ids = torch.cat(negative_input_ids, dim=0)

    anchor_attention_masks = torch.cat(anchor_attention_masks, dim=0)
    positive_attention_masks = torch.cat(positive_attention_masks, dim=0)
    negative_attention_masks = torch.cat(negative_attention_masks, dim=0)

    user_infos = torch.tensor(user_infos, dtype=torch.int64)
    sev_labs = torch.tensor(sev_labs, dtype = torch.int64)
    st_labs = torch.tensor(st_labs, dtype=torch.int64)
    rebut_labs = torch.tensor(rebut_labs, dtype=torch.int64)
    topic_labs = torch.tensor(topic_labs, dtype=torch.int64)

    return anchor_input_ids, positive_input_ids, negative_input_ids,\
           anchor_attention_masks, positive_attention_masks, negative_attention_masks,\
           user_infos, sev_labs, st_labs, rebut_labs, topic_labs

def sort_on_topic(tweet_anno_dict):
    topic_dict = {}

    for id in tweet_anno_dict.keys():
        topic = tweet_anno_dict[id]['topic']
        if topic not in topic_dict.keys():
            topic_dict[topic] = []

        topic_dict[topic].append(id)

    temp_list = []
    for topic in topic_dict.keys():
        temp_list.append(topic_dict[topic])


    temp_list.sort(key=len, reverse=True)
    sorted_ids = []
    for list in temp_list:
        sorted_ids.extend(list)

    return sorted_ids

def parse_and_tensify(tweet_anno_dict, tweet_dict):
    ### GROUP THE DATA WE ACTUALLY WANT TOGETHER ###
    # First thing, take all of the annotated files and find the values we want
    # Sentences, User information (as list), labels

    print(f"Sorting on topics.")
    ids_list = sort_on_topic(tweet_anno_dict)
    length = len(ids_list)
    tr = int(.7 * length)
    val = int(.1 * length)


    ### TOKENIZATION GOODNESS ###
    # -------------------------------#
    print("Tokenizing. ")
    # -------------------------------#
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 200  # PLACEHOLDER
    tweet_anno_dict = encode_sentences(tweet_anno_dict, tweet_dict, tokenizer, max_len)

    tr_ids_list = ids_list[:tr]
    val_ids_list = ids_list[tr:]

    print(f"Length of tr_ids_list: {len(tr_ids_list)}")
    print(f"Sampling on training topics.")
    tr_triplets = zs.sample_topics(tr_ids_list, tweet_anno_dict)
    print(f"Length of triplets: {len(tr_triplets)}")

    print(f"Sampling on validation topics.")
    val_triplets = zs.sample_topics(val_ids_list, tweet_anno_dict)


    print(f"Retrieving training annotation ids.")
    tr_data = retrieve_anno_ids(tweet_anno_dict, tweet_dict, tr_triplets)
    tr_triplet_input_ids, tr_triplet_attention_masks, tr_user_infos, \
        tr_sev_labs, tr_st_labs, tr_rebut_labs, tr_topic_labs = tr_data

    # Shuffle
    d = list(zip(tr_triplet_input_ids, tr_triplet_attention_masks, tr_user_infos,
                 tr_sev_labs, tr_st_labs, tr_rebut_labs, tr_topic_labs))
    random.shuffle(d)
    tr_triplet_input_ids, tr_triplet_attention_masks, tr_user_infos, \
        tr_sev_labs, tr_st_labs, tr_rebut_labs, tr_topic_labs = zip(*d)

    print(f"Retrieving validation annotation ids.")
    val_data = retrieve_anno_ids(tweet_anno_dict, tweet_dict, val_triplets)
    val_triplet_input_ids, val_triplet_attention_masks, val_user_infos, \
        val_sev_labs, val_st_labs, val_rebut_labs, val_topic_labs = val_data







    # TOKEN IDS #
    # -------------------------------#
    print("Tensifying. ")
    # -------------------------------#
    training_data = ret_ids_from_triplets(tr_triplet_input_ids, tr_triplet_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs, tr_topic_labs,
                                             tokenizer,
                                             maximum_length=max_len
                                             )

    validation_data = ret_ids_from_triplets(val_triplet_input_ids, val_triplet_attention_masks, val_user_infos, val_sev_labs, val_st_labs, val_rebut_labs, val_topic_labs,
                                               tokenizer,
                                               maximum_length=max_len
                                               )

    return training_data, validation_data


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))
def apn_skl(y_true, y_pred, list_type):

    p, r, f, s = met.precision_recall_fscore_support(y_true, y_pred, average=None)
    mp, mr, m, ms = met.precision_recall_fscore_support(y_true, y_pred, average='micro')
    print(f'============================')
    print(f'For --{list_type}--\nPrecision: {p}\nRecall: {r}\nF1: {f}\nMicro: {m}')
    #print(f'\n=========================\nF1 for {list_type} via scikit: {p}, {r}, {f}\n======================\n')
    pass

# list_type: accuracy, positives, negatives
def apn(y_true, y_pred, list_type):
    if list_type == 'severity':
        use_list = severity_list
    elif list_type == 'stance':
        use_list = stance_list
    elif list_type == 'rebuttal':
        use_list = rebuttal_list

    accuracy = 0

    TP = np.zeros(len(use_list))
    FP = np.zeros(len(use_list))
    TN = np.zeros(len(use_list))
    FN = np.zeros(len(use_list))

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        for j, c in enumerate(use_list):
            if t == j:
                TP[j] += t == p
                FN[j] += t != p
            elif p == j:
                FP[j] += p != t
            elif p != j and t != j:
                TN[j] += 1
        accuracy += (t == p)

    accuracy = accuracy / len(y_true)
    return accuracy, TP, FP, TN, FN

# Add batched tp to total tp, etc
def sum_pn(ttp, tfp, ttn, tfn, tp, fp, tn, fn):
    return ttp + tp, tfp + fp, ttn + tn, tfn + fn

# Metrics maybe
def pre_rec_f1(acc, tp, fp, tn, fn, type=None):
    accuracy = acc

    tpfp = tp + fp
    tpfn = tp + fn
    precision = np.divide(tp, tpfp, out=np.zeros_like(tp), where=tpfp!=0)
    recall = np.divide(tp, tpfn, out=np.zeros_like(tp), where=tpfn!=0)
    #precision = np.nan_to_num(tp / (tp + fp), nan=0)
    #recall = np.nan_to_num(tp / (tp + fn), nan=0)
    PR2 = 2 * precision * recall
    PORR = precision + recall
    f1 = np.divide(PR2, PORR, out=np.zeros_like(PR2), where=PORR!=0)
    #f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))

    if type == None:
        print(f'Accuracy: {accuracy}\n',
              f'Precision: {precision}\n',
              f'Recall: {recall}\n',
              f'F1 Score: {f1}\n')
    elif type=='severity':
        print(f'Accuracy: {accuracy}\n',
              f'Precision: {precision}\n',
              f'Recall: {recall}\n')
        for sev, f in zip(severity_list, f1):
            print(f'{sev:20}: {f:.4f}')
    elif type=='stance':
        print(f'Accuracy: {accuracy}\n',
              f'Precision: {precision}\n',
              f'Recall: {recall}\n')
        for st, f in zip(stance_list, f1):
            print(f'{st:20}: {f:.4f}')
    elif type=='rebuttal':
        print(f'Accuracy: {accuracy}\n',
              f'Precision: {precision}\n',
              f'Recall: {recall}\n')
        for reb, f in zip(rebuttal_list, f1):
            print(f'{reb:20}: {f:.4f}')
    print('~~~'*15)


    return accuracy, precision, recall, f1