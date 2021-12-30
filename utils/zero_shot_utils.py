# Given total number of points, pull examples for a tweet from both positive and negative.
import random
import math
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import torch
import numpy as np
import sklearn.metrics as met
import sklearn.multiclass
from transformers import BertTokenizer


def sample_topics(tweet_ids, anno_dict=None):
    """
    Create a list for each topic
    Sample from list for positive and negative examples
    :param anno_dict:
    :return:

    Assumption
    anno_dict[id]['topic'] = topic
    """

    #random.seed(34)
    topics_dict = {}

    print(len(tweet_ids))
    for tweet_id in tweet_ids:
        if tweet_id in anno_dict.keys():
            topic = anno_dict[tweet_id]['topic']
            if topic != 0:
                if topic not in topics_dict.keys():
                    topics_dict[topic] = []
                topics_dict[topic].append(tweet_id)


    topics = topics_dict.keys()
    '''
    # For each topic, grab a positive example and a negative example
    # (anchor, positive, negative)
    # t1, t2, t3, ... ,tn
    # from ta grab t1_i, t1_j, where j is not i
    # grab b s.t. tb != ta for some sample of b
    # sample from both
    '''
    duped_data = set()
    print(f"topics: {sorted(list(topics_dict.keys()))}")
    for topic in tqdm(topics):
        anchor_divisor = 10  # get 1/anchor_divisor number of anchors
        # sample anchors:
        anchors = random.sample(topics_dict[topic], math.ceil((len(topics_dict[topic])/anchor_divisor)))
        anchor_set = set()
        for anchor in anchors:
            anchor_set.add(anchor)

        for anchor in anchors:
            neg_topic_divisor = 3  # get 1/neg_topic_divisor negative topics as negative example for this anchor
            # Sample negative topics
            neg_topics = []
            neg_topics_set = set(topics)
            neg_topics_set.remove(topic)
            while len(neg_topics) < math.ceil((len(topics)/neg_topic_divisor)):
                nt_to_add = random.sample(neg_topics_set, 1)
                neg_topics_set.remove(nt_to_add[0])
                if nt_to_add[0] != topic and nt_to_add[0] not in neg_topics:
                    neg_topics.append(nt_to_add[0])
            # print(f'\nFor topic {topic}, we will sample from these negative topics:')
            # for nt in neg_topics:
            #     print(f'{nt}', end=' ')

            positive_examples = sample_post_from_topic(topics_dict, topic, anchor)

            negative_examples = []
            for nt in neg_topics:
                negative_examples.extend(sample_post_from_topic(topics_dict, nt, anchor))

            combinations = merge_combination(anchor, positive_examples, negative_examples)
            for combination in combinations:
                duped_data.add(combination)

    dupeless_data = remove_duplications(duped_data)
    print(f"LENGTH OF DUPELESS DATA: {len(dupeless_data)}")
    return dupeless_data
    # print(f'Length of data is: {len(data)}.')
    # for d in data:
    #     print('-'*30)
    #     print(d)
    #     a, p, n = d[0], d[1], d[2]
    #     print(f"{anno_dict[a]['topic']}-{anno_dict[p]['topic']}-{anno_dict[n]['topic']}")

def sample_post_from_topic(topics_dict, topic, anchor):
    """
    Given a topic and anchor, sample roughly 1/4 of the topics from the topic list s.t. they are not equal to anchor
    and there are no duplicates
    :param topics_dict: Dictionary of the form topics_dict[topic] = list of ids
    :param topic: Topic to access list of ids
    :param anchor: The anchor id for positive examples
    :return:
    """

    example_divisor = 4  # Grab 1/example_divisor number of examples (pos/neg)
    example = []
    # print(f"--- Sampling from topic {topic}. ---")
    example_set = set(topics_dict[topic])
    while (len(example) < math.ceil((len(topics_dict[topic])/example_divisor))) and len(topics_dict[topic]) > 1:
        ex_to_add = random.sample(example_set, 1)
        if ex_to_add[0] != anchor and ex_to_add[0] not in example:
            example_set.remove(ex_to_add[0])
            example.append(ex_to_add[0])

    return example

def merge_combination(anchor, positive_examples, negative_examples):
    combinations = []
    # print(f"--- Merging positive and negative examples. ---")
    for pe in positive_examples:
        for ne in negative_examples:
            combinations.append((anchor, pe, ne))

    return combinations


def remove_duplications(combinations):
    """
    :param combinations: a list of combinations
    :return:
    combination = [anchor, positive_example, negative_example]
    """
    total_examples = set()
    # print(f"--- Removing duplicates. ---")
    for combination in combinations:
        anchor = combination[0]
        positive_example = combination[1]
        negative_example = combination[2]

        if (positive_example, anchor, negative_example) not in total_examples:
            total_examples.add((anchor, positive_example, negative_example))

    return total_examples

def create_data_dicts(anno_dict, tweet_dict):
    """
    This is just a complete dictionary of everything. This way you can always
    grab the data you need by passing in the id to anno_dict
    :param anno_dict:
    :param tweet_dict:
    :return:
    """
    complete_data = {}

    location_emb = {}

    user_infos = []
    for id in anno_dict.keys():
        complete_data[id] = {}
        if id in tweet_dict.keys():
            if tweet_dict[id]['user_loc'] not in location_emb.keys():
                location_emb[tweet_dict[id]['user_loc']] = len(location_emb)
            fol_c = int(tweet_dict[id]['user_follower_count'])
            fr_c = int(tweet_dict[id]['user_friend_count'])
            fav_c = int(tweet_dict[id]['user_fav_count'])
            st_c = int(tweet_dict[id]['user_statuses_count'])

            time_diff = datetime.strptime(tweet_dict[id]['created_at'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(
                tweet_dict[id]['user_created'], '%Y-%m-%d %H:%M:%S')

            cr = int(time_diff.total_seconds())
            ver = int(tweet_dict[id]['user_verif'])
            # loc = location_emb[tweet_dict[id]['user_loc']]

            ### GRAB USER INFOS ###
            user_info = [fol_c, fr_c, fav_c, st_c, cr, ver]
            user_infos.append(user_info) # Create this so that we can normalize each thing.
            # complete_data[id]['user_info'] = user_info


            ### GRAB SENTENCE ###
            complete_data[id]['input_ids'] = anno_dict[id]['input_ids']
            complete_data[id]['attention_mask'] = anno_dict[id]['attention_mask']

            ### GRAB SEVERITY ###
            complete_data[id]['severity'] = anno_dict[id]['severity']

            ### GRAB STANCE ###
            complete_data[id]['stance'] = anno_dict[id]['stance']

            ### GRAB REBUTTAL ###
            complete_data[id]['rebuttal'] = anno_dict[id]['rebuttal']

            ### GRAB TOPIC ###
            complete_data[id]['topic'] = anno_dict[id]['topic']

            complete_data[id]['theme'] = anno_dict[id]['theme']

    user_infos_np = np.array(user_infos)
    ui_mins = np.amin(user_infos_np, 0)
    ui_maxs = np.max(user_infos_np, 0)

    # Update the user info
    for id in anno_dict.keys():
        if tweet_dict[id]['user_loc'] not in location_emb.keys():
            location_emb[tweet_dict[id]['user_loc']] = len(location_emb)
        fol_c = int(tweet_dict[id]['user_follower_count'])
        fr_c = int(tweet_dict[id]['user_friend_count'])
        fav_c = int(tweet_dict[id]['user_fav_count'])
        st_c = int(tweet_dict[id]['user_statuses_count'])

        time_diff = datetime.strptime(tweet_dict[id]['created_at'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(
            tweet_dict[id]['user_created'], '%Y-%m-%d %H:%M:%S')

        cr = int(time_diff.total_seconds())
        ver = int(tweet_dict[id]['user_verif'])
        loc = location_emb[tweet_dict[id]['user_loc']]

        user_info = np.array([fol_c, fr_c, fav_c, st_c, cr, ver])
        user_info = (user_info - ui_mins) / (ui_maxs - ui_mins)
        user_info = list(user_info)
        user_info.append(loc)

        complete_data[id]['user_info'] = user_info

    return complete_data
# if __name__ == '__main__':
#     test_dict = {}
#     for i in range(20):
#         test_dict[i] = {}
#         test_dict[i]['topic'] = random.randint(0, 6)
#
#     data = sample_topics(test_dict)

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

def retrieve_id_splits(tweet_anno_dict, seedval):
    topics_list = set()
    tr_topics, val_topics, te_topics = set(), set(), set()
    random.seed(seedval)
    # TOPIC ID'S FOR EACH SEVERITY
    not_severe_topics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17,
                         31, 34, 36, 38, 44]
    possibly_severe_topics = [18,
                              21, 22, 23, 27, 28,
                              32, 41, 42, 45]
    highly_severe_topics = [20, 24, 25, 26, 29,
                            30, 33, 35, 37, 39,
                            40, 43]

    posshigh = possibly_severe_topics + highly_severe_topics

    real_topics = [19,
                   61, 62, 63, 64, 65, 66, 67, 68, 69]

    #sev_topics = [not_severe_topics, possibly_severe_topics, highly_severe_topics, real_topics]
    sev_topics = [not_severe_topics, posshigh, real_topics]

    specific_topic_dict = {}
    #tops_list = [x for topic in sev_topics for x in topic]
    for id in tweet_anno_dict.keys():
        if tweet_anno_dict[id]['topic'] not in specific_topic_dict.keys():
            specific_topic_dict[tweet_anno_dict[id]['topic']] = []
        specific_topic_dict[tweet_anno_dict[id]['topic']].append(id)

    extra_tr_topics = []
    topics_to_remove = []
    for setops in sev_topics:
        for tpid in setops:
            if tpid not in specific_topic_dict.keys() or len(specific_topic_dict[tpid]) <= 5:
                topics_to_remove.append(tpid)
                if tpid in specific_topic_dict.keys() and len(specific_topic_dict[tpid]) > 2:
                    extra_tr_topics.append(tpid)


    # for topic in specific_topic_dict.keys():
    #     if len(specific_topic_dict[topic]) < 4:
    #         topics_to_remove.append(topic)
    #         if len(specific_topic_dict[topic]) > 2:
    #             extra_tr_topics.append(topic)
    #         #specific_topic_dict.pop(topic, None)

    # Any topic with less than 4 ids, remove it
    for setops in sev_topics:
        for rmtop in topics_to_remove:
            if rmtop in setops:
                #print(f'popping topic {rmtop}')
                setops.remove(rmtop)
    # for topic in sev_topics:
    #     for rmtop in topics_to_remove:
    #         if rmtop in topic:
    #
    #             topic.remove(rmtop)


    tr_topics, val_topics, te_topics, trval_topics = set(), set(), set(), set()
    for se_to in sev_topics:
        tmp_topics = set(se_to.copy())

        # changed from 2 to 3 for posshigh
        val_top = set(random.sample(tmp_topics, 3))
        val_topics.update(val_top)
        tmp_topics = tmp_topics - val_top

        test_top = set(random.sample(tmp_topics, 3))
        te_topics.update(test_top)
        tmp_topics = tmp_topics - test_top

        trval_topics = set(random.sample(tmp_topics, 1))
        tmp_topics = tmp_topics - trval_topics
        tr_topics.update(tmp_topics)

    tr_topics.update(extra_tr_topics)
    #print(f"Val_sizes = {[specific_topic_dict[x] for x in val_topics if x in specific_topic_dict.keys()]}")
    #tr_topics.update(extra_tr_topics)
    # for id in tweet_anno_dict.keys():
    #     topic = tweet_anno_dict[id]['topic']
    #     topics_list.add(topic)
    # print(f"Length of topics: {len(topics_list)}")
    #
    # total_size = len(topics_list)
    # training_size = int(.75 * total_size)
    # val_size = int(.15 * total_size)
    # test_size = int(.25 * total_size)
    #
    # # Take 70% of the topics as training
    # tr_topics = set(random.sample(topics_list, training_size))
    # # Make the test set the remaining 30%
    # te_topics = topics_list.copy() - tr_topics.copy()
    # # Grab 10% of total from the test set, remove from test set
    # val_unique = set(random.sample(te_topics, val_size))
    # # Remove uniques from the test_topics
    # te_topics = te_topics - val_unique
    # # Grab 5% of the training, do not remove from training (overlap)
    # val_shared = set(random.sample(tr_topics, int(val_size/2)))
    # # Add the uniques and the shared together to make the validation set
    # val_topics = val_unique.union(val_shared)#val_unique + val_shared #val_unique.add(val_shared)
    print(f"Topics - {tr_topics} - {val_topics} - {te_topics}")



    training_ids, validation_ids, test_ids, trval_ids = set(), set(), set(), set()
    for id in tweet_anno_dict.keys():
        topic = tweet_anno_dict[id]['topic']
        if topic in tr_topics:
            training_ids.add(id)
        if topic in val_topics:
            validation_ids.add(id)
        if topic in te_topics:
            test_ids.add(id)
        if topic in trval_topics:
            trval_ids.add(id)


    val_sep_ids = set(random.sample(trval_ids, int(0.2 * len(trval_ids))))
    validation_ids.update(val_sep_ids)
    training_ids.update((trval_ids - val_sep_ids))

    # tmp = validation_ids.copy()
    # validation_ids = test_ids.copy()
    # test_ids = tmp

    return list(training_ids), \
           list(validation_ids), \
           list(test_ids)

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

def return_topics_dict_given_id(id_list, tweet_anno_dict):
    specific_topic_dict = {}
    ids_to_remove = []
    for id in id_list:
        if tweet_anno_dict[id]['topic'] not in specific_topic_dict.keys():
            specific_topic_dict[tweet_anno_dict[id]['topic']] = []
        specific_topic_dict[tweet_anno_dict[id]['topic']].append(id)

    topics_to_remove = []
    for topic in specific_topic_dict.keys():
        if len(specific_topic_dict[topic]) < 4:
            #specific_topic_dict.pop(topic, None)
            topics_to_remove.append(topic)

    for topic in topics_to_remove:
        if topic in specific_topic_dict.keys():
            del specific_topic_dict[topic]

    for id in id_list:
        if tweet_anno_dict[id]['topic'] in topics_to_remove:
            id_list.remove(id)

    return specific_topic_dict, id_list

def parse_training_validation_test(tweet_anno_dict, tweet_dict, seed_val):
    # ids_list = sort_on_topic(tweet_anno_dict)
    # length = len(ids_list)
    # tr = int(.80 * length)
    # #tr1 = int(.4 * length)
    # tr2 = int(.13 * length)
    #
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # max_len = 200  # PLACEHOLDER
    # tweet_anno_dict = encode_sentences(tweet_anno_dict, tweet_dict, tokenizer, max_len)
    # completed_data_dict = create_data_dicts(tweet_anno_dict, tweet_dict)
    #
    #
    # ### YOLO TESTING SOMETHING ###
    # #random.shuffle(ids_list)
    # #training_valid = ids_list[:tr]
    # #test_ids_list = ids_list[tr:]
    # tv_part = ids_list[:tr]
    # random.shuffle(tv_part)
    #
    #
    # test_ids_list_part = tv_part[int(.81 * len(tv_part)):]
    #
    # test_ids_list = ids_list[tr + tr2:] + test_ids_list_part
    # training_valid = tv_part[:int(.81 *len(tv_part))]
    # #training_valid = ids_list[tr2:-tr2]
    # #test_ids_list = ids_list[:tr2] + ids_list[-tr2:]
    #
    #
    # #shuffle training ids:
    # #random.shuffle(training_valid)
    # # create validation
    # # 90%
    #
    # tr_v = int(.85 * len(training_valid))
    # #tr_v_1 = int(.45 * len(training_valid))
    # #tr_v_2 = int(.05 * len(training_valid))
    # tr_ids_list = training_valid[:tr_v]
    # val_ids_list = training_valid[tr_v:]
    #
    # #tr_ids_list = training_valid[:tr_v_1] + training_valid[tr_v_1 + tr_v_2:]
    # #val_ids_list = training_valid[tr_v_1:tr_v_1+tr_v_2] + training_valid[tr_v_1+tr_v_1:]
    #----------------------------------------------
    tr_ids_list, val_ids_list, te_ids_list = retrieve_id_splits(tweet_anno_dict, seed_val)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 200  # PLACEHOLDER
    tweet_anno_dict = encode_sentences(tweet_anno_dict, tweet_dict, tokenizer, max_len)
    completed_data_dict = create_data_dicts(tweet_anno_dict, tweet_dict)

    training_topic_dict,   tr_ids_list  = return_topics_dict_given_id(tr_ids_list,  completed_data_dict)
    validation_topic_dict, val_ids_list = return_topics_dict_given_id(val_ids_list, completed_data_dict)
    testing_topic_dict,    te_ids_list  = return_topics_dict_given_id(te_ids_list,  completed_data_dict)
    #validation_topic_dict = return_topics_dict_given_id(te_ids_list, completed_data_dict)


    tr_ids_list = torch.tensor(tr_ids_list, dtype=torch.int64)
    val_ids_list = torch.tensor(val_ids_list, dtype=torch.int64)
    te_ids_list = torch.tensor(te_ids_list, dtype=torch.int64)
    #testing_topic_dict, test_ids_list = None, None

    return tr_ids_list, training_topic_dict, \
           val_ids_list, validation_topic_dict, \
           te_ids_list, testing_topic_dict, completed_data_dict

def sample_topics_online(batch, complete_data_dict, topics_dict):
    """

    :param batch:
    :param complete_data_dict:
    :param topics_dict: Should have a different one for both training and validation
    :return:
    """
    b_a_iids = []
    b_a_mask = []
    b_p_iids = []
    b_p_mask = []
    b_n_iids = []
    b_n_mask = []

    b_user_infos = []
    b_sev_labels = []
    b_st_labels = []
    b_rebut_labels = []
    b_topic_labels = []

    topics = set(topics_dict.keys())
    #print(f"Batch: {batch}")
    for id in batch[0].tolist(): # Maybe batch[0]? Gotta test
        a_iids = complete_data_dict[id]["input_ids"]
        a_mask = complete_data_dict[id]["attention_mask"]

        user_infos = complete_data_dict[id]["user_info"]
        sev_label = complete_data_dict[id]["severity"]
        st_label = complete_data_dict[id]["stance"]
        rebut_label = complete_data_dict[id]["rebuttal"]
        topic_label = complete_data_dict[id]["topic"]

        similar_posts = set(topics_dict[topic_label].copy())
        if id in similar_posts:
            similar_posts.remove(id)
        else:
            print('skipped')
        positive_id = random.sample(similar_posts, 1)

        temp_topics = topics.copy()
        temp_topics.remove(topic_label)
        negative_topic = random.sample(temp_topics, 1)
        negative_id = random.sample(topics_dict[negative_topic[0]], 1)

        # - - - - - - - - - - - - - - - - - - #
        # print(f"Triplet: ({id}, {positive_id[0]}, {negative_id[0]})")
        # - - - - - - - - - - - - - - - - - - #

        p_iids = complete_data_dict[positive_id[0]]["input_ids"]
        p_mask = complete_data_dict[positive_id[0]]["attention_mask"]

        n_iids = complete_data_dict[negative_id[0]]["input_ids"]
        n_mask = complete_data_dict[negative_id[0]]["attention_mask"]

        b_a_iids.append(a_iids)
        b_a_mask.append(a_mask)
        b_p_iids.append(p_iids)
        b_p_mask.append(p_mask)
        b_n_iids.append(n_iids)
        b_n_mask.append(n_mask)
        b_user_infos.append(user_infos)
        b_sev_labels.append(sev_label)
        b_st_labels.append(st_label)
        b_rebut_labels.append(rebut_label)
        b_topic_labels.append(topic_label)

    # Create them as tensors #
    b_a_iids = torch.cat(b_a_iids, dim=0)
    b_a_mask = torch.cat(b_a_mask, dim=0)
    b_p_iids = torch.cat(b_p_iids, dim=0)
    b_p_mask = torch.cat(b_p_mask, dim=0)
    b_n_iids = torch.cat(b_n_iids, dim=0)
    b_n_mask = torch.cat(b_n_mask, dim=0)
    b_user_infos = torch.tensor(b_user_infos, dtype=torch.int64)
    b_sev_labels = torch.tensor(b_sev_labels, dtype=torch.int64)
    b_st_labels = torch.tensor(b_st_labels, dtype=torch.int64)
    b_rebut_labels = torch.tensor(b_rebut_labels, dtype=torch.int64)
    b_topic_labels = torch.tensor(b_topic_labels, dtype=torch.int64)

    return b_a_iids, b_p_iids, \
            b_n_iids, b_a_mask, \
            b_p_mask, b_n_mask, \
            b_user_infos, b_sev_labels, b_st_labels, b_rebut_labels, b_topic_labels
