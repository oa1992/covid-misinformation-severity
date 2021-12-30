import utils.data_utils as du
import utils.zero_shot_utils as zsu
import time
import models.BertUserModel as bum
import models.sestre as ses
import torch

def retrieve_data(batch, complete_data_dict, topics_dict):
    """

    :param batch:
    :param complete_data_dict:
    :param topics_dict: Should have a different one for both training and validation
    :return:
    """
    b_iids = []
    b_masks = []

    b_user_infos = []
    b_sev_labels = []
    b_st_labels = []
    b_rebut_labels = []
    b_topic_labels = []
    b_theme_labels = []

    
    for id in batch[0].tolist(): # Maybe batch[0]? Gotta test
        iids = complete_data_dict[id]["input_ids"]
        mask = complete_data_dict[id]["attention_mask"]

        user_infos = complete_data_dict[id]["user_info"]
        sev_label = complete_data_dict[id]["severity"]
        st_label = complete_data_dict[id]["stance"]
        rebut_label = complete_data_dict[id]["rebuttal"]
        topic_label = complete_data_dict[id]["topic"]
        theme_label = complete_data_dict[id]["theme"]

        
        b_iids.append(iids)
        b_masks.append(mask)

        b_user_infos.append(user_infos)
        b_sev_labels.append(sev_label)
        b_st_labels.append(st_label)
        b_rebut_labels.append(rebut_label)
        b_topic_labels.append(topic_label)
        b_theme_labels.append(theme_label)

    # Create them as tensors #
    b_iids = torch.cat(b_iids, dim=0)
    b_masks = torch.cat(b_masks, dim=0)
    
    b_user_infos = torch.tensor(b_user_infos, dtype=torch.int64)
    b_sev_labels = torch.tensor(b_sev_labels, dtype=torch.int64)
    b_st_labels = torch.tensor(b_st_labels, dtype=torch.int64)
    b_rebut_labels = torch.tensor(b_rebut_labels, dtype=torch.int64)
    b_topic_labels = torch.tensor(b_topic_labels, dtype=torch.int64)
    b_theme_labels = torch.tensor(b_theme_labels, dtype=torch.int64)

    return b_iids, b_masks, \
            b_user_infos, b_sev_labels, b_st_labels, b_rebut_labels, b_topic_labels, b_theme_labels
def inner_sestre(batch, device, model, complete_data_dict, topics_dict, topic_embeddings, type="train"):
    b_data = retrieve_data(batch, complete_data_dict, topics_dict)

    b_input_ids = b_data[0].to(device)
    b_masks = b_data[1].to(device)
    b_user_infos = b_data[2].to(device)
    b_sev_labels = b_data[3].to(device)
    b_st_labels = b_data[4].to(device)
    b_rebut_labels = b_data[5].to(device)
    b_topic_labels = b_data[6].to(device)
    b_theme_labels = b_data[7].to(device)

    batch_size = len(batch[0])
    # print(batch)
    if type == 'train':
        model.zero_grad()
        sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_sestre(b_data, device, model,
                                                                                       batch_size, topic_embeddings)
        return total_loss, model

    elif type == 'val':
        with torch.no_grad():
            sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_sestre(b_data, device, model,
                                                                                           batch_size, topic_embeddings)

            # Severity test scikit
            sev_preds = sev_preds.detach().cpu().numpy()
            sev_labels = b_sev_labels.to('cpu').numpy()

            st_preds = st_preds.detach().cpu().numpy()
            st_labels = b_st_labels.to('cpu').numpy()

            rebut_preds = rebut_preds.detach().cpu().numpy()
            rebut_labels = b_rebut_labels.to('cpu').numpy()

            #theme_preds = theme_preds.detach().cpu().numpy()
            #theme_labels = b_theme_labels.to('cpu').numpy()

            return total_loss, model, \
                   sev_preds, sev_labels, \
                   st_preds, st_labels, \
                   rebut_preds, rebut_labels#, \
                    #theme_preds, theme_labels

def retrieve_preds_sestre(b_data, device, model, batch_size, topic_embeddings):

    b_input_ids = b_data[0].to(device)
    b_masks = b_data[1].to(device)
    b_user_infos = b_data[2].to(device)
    b_sev_labels = b_data[3].to(device)
    b_st_labels = b_data[4].to(device)
    b_rebut_labels = b_data[5].to(device)
    b_topic_labels = b_data[6].to(device)
    b_theme_labels = b_data[7].to(device)
    #print(topic_embeddings)
    sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_masks,
                                                         topic_embedding=topic_embeddings,
                                                         user_infos=b_user_infos,
                                                         sev_labels=b_sev_labels, st_labels=b_st_labels, rebut_labels=b_rebut_labels,
                                                         topic_labels=b_topic_labels, theme_labels=b_theme_labels,
                                                         batch_size=batch_size)

    return sev_preds, st_preds, rebut_preds, total_loss, model