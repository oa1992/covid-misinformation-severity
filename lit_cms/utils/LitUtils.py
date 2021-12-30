import torch
from transformers import BertTokenizer

# def retrieve_data(tweet_ids, complete_data_dict, topics_dict):
#     """
#
#     :param batch:
#     :param complete_data_dict:
#     :param topics_dict: Should have a different one for both training and validation
#     :return:
#     """
#     b_iids = []
#     b_masks = []
#
#     b_user_infos = []
#     b_sev_labels = []
#     b_st_labels = []
#     b_rebut_labels = []
#     b_topic_labels = []
#     b_theme_labels = []
#     for tid in tweet_ids:#[0].tolist():
#
#         id = tid.item()
#         iids = complete_data_dict[id]["input_ids"]
#         mask = complete_data_dict[id]["attention_mask"]
#
#         user_infos = complete_data_dict[id]["user_info"]
#         sev_label = complete_data_dict[id]["severity"]
#         st_label = complete_data_dict[id]["stance"]
#         rebut_label = complete_data_dict[id]["rebuttal"]
#         topic_label = complete_data_dict[id]["topic"]
#         theme_label = complete_data_dict[id]["theme"]
#
#         tweet_id = tid
#
#         b_iids.append(iids)
#         b_masks.append(mask)
#
#         b_user_infos.append(user_infos)
#         b_sev_labels.append(sev_label)
#         b_st_labels.append(st_label)
#         b_rebut_labels.append(rebut_label)
#         b_topic_labels.append(topic_label)
#         b_theme_labels.append(theme_label)
#
#     # Create them as tensors #
#     b_iids = torch.cat(b_iids, dim=0)
#     b_masks = torch.cat(b_masks, dim=0)
#
#     #b_iids = torch.tensor(b_iids, dtype=torch.int64)
#     #b_masks = torch.tensor(b_masks, dtype=torch.int64)
#     b_user_infos = torch.tensor(b_user_infos, dtype=torch.int64)
#     b_sev_labels = torch.tensor(b_sev_labels, dtype=torch.int64)
#     b_st_labels = torch.tensor(b_st_labels, dtype=torch.int64)
#     b_rebut_labels = torch.tensor(b_rebut_labels, dtype=torch.int64)
#     b_topic_labels = torch.tensor(b_topic_labels, dtype=torch.int64)
#     b_theme_labels = torch.tensor(b_theme_labels, dtype=torch.int64)
#
#     print(b_iids.size(), b_masks.size())
#
#     return b_iids, b_masks, \
#            b_user_infos, b_sev_labels, b_st_labels, b_rebut_labels, b_topic_labels, b_theme_labels


def retrieve_data(tweet_ids, complete_data_dict, topics_dict):
    """

    :param batch:
    :param complete_data_dict:
    :param topics_dict: Should have a different one for both training and validation
    :return:
    """
    b_ids = []
    b_iids = []
    b_masks = []

    b_user_infos = []
    b_sev_labels = []
    b_st_labels = []
    b_rebut_labels = []
    b_topic_labels = []
    b_theme_labels = []
    for tid in tweet_ids:  # [0].tolist():

        id = tid.item()
        iids = complete_data_dict[id]["input_ids"]
        mask = complete_data_dict[id]["attention_mask"]

        user_infos = complete_data_dict[id]["user_info"]
        sev_label = complete_data_dict[id]["severity"]
        st_label = complete_data_dict[id]["stance"]
        rebut_label = complete_data_dict[id]["rebuttal"]
        topic_label = complete_data_dict[id]["topic"]
        theme_label = complete_data_dict[id]["theme"]

        #print(iids)
        b_ids.append(id)

        b_iids.append(iids)
        b_masks.append(mask)

        b_user_infos.append(user_infos)
        b_sev_labels.append(sev_label)
        b_st_labels.append(st_label)
        b_rebut_labels.append(rebut_label)
        b_topic_labels.append(topic_label)
        b_theme_labels.append(theme_label)

    # Create them as tensors #
    b_ids = torch.tensor(b_ids, dtype=torch.int64)
    b_iids = torch.cat(b_iids, dim=0)
    b_masks = torch.cat(b_masks, dim=0)

    # b_iids = torch.tensor(b_iids, dtype=torch.int64)
    # b_masks = torch.tensor(b_masks, dtype=torch.int64)
    b_user_infos = torch.tensor(b_user_infos, dtype=torch.int64)
    b_sev_labels = torch.tensor(b_sev_labels, dtype=torch.int64)
    b_st_labels = torch.tensor(b_st_labels, dtype=torch.int64)
    b_rebut_labels = torch.tensor(b_rebut_labels, dtype=torch.int64)
    b_topic_labels = torch.tensor(b_topic_labels, dtype=torch.int64)
    b_theme_labels = torch.tensor(b_theme_labels, dtype=torch.int64)

    print(b_iids.size(), b_masks.size())

    return {"input_ids": b_iids,
            "masks": b_masks,
            "user_infos": b_user_infos,
            "severity_labels": b_sev_labels, "stance_labels": b_st_labels, "rebuttal_labels": b_rebut_labels,
            "topic_labels": b_topic_labels, "theme_labels": b_theme_labels,
            "tweet_ids": b_ids}

