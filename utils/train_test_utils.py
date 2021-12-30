import utils.data_utils as du
import utils.zero_shot_utils as zsu
import time
import models.BertUserModel as bum
import models.sestre as ses
import torch

def traintest(dataloader, model, type='train'):
    '''

    :param dataloader: The training/validation/test dataloader
    :param model: the model we are currently using
    :param type:
    :return:
    '''
    # Progress update every 10 batches.

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Record time, get time ready.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0

    if type=='train':
        model.train()
    elif type=='eval':
        model.eval()


    for step, batch in enumerate(dataloader):
        if step % 10 == 0 and not step == 0:
            elapsed = du.format_time(time.time() - t0)
            print(f'Batch{step:>5,} of {len(dataloader):>5,}. Elapsed: {elapsed}')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_user_infos = batch[2].to(device)
        b_sev_labels = batch[3].to(device)
        b_st_labels = batch[4].to(device)
        b_rebut_labels = batch[5].to(device)

        ### ROUND 1 - Not using user_infos ###
        model.zero_grad()

        sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids,
                                                             token_type_ids=None,
                                                             attention_mask=b_input_mask,
                                                             user_infos=b_user_infos,
                                                             sev_labels=b_sev_labels,
                                                             st_labels=b_st_labels,
                                                             rebut_labels=b_rebut_labels)

        total_train_loss += total_loss.item()
        total_loss.backward()

    avg_train_loss = total_train_loss / len(dataloader)
    training_time = du.format_time(time.time() - t0)

    print("")
    print(f" Average training loss: {avg_train_loss:.2f}")
    print(f" Training epoch took: {training_time}")

    print("")
    print("Running Validation...")

def inner(batch, device, model, type='train'):
    # tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_user_infos = batch[2].to(device)
    b_sev_labels = batch[3].to(device)
    b_st_labels = batch[4].to(device)
    b_rebut_labels = batch[5].to(device)
    b_topic_labels = batch[6].to(device)

    if type=='train':
        model.zero_grad()
        sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds(batch, device, model)
        return total_loss, model

    elif type=='val':
        with torch.no_grad():
            sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds(batch, device, model)

            # Severity test scikit
            sev_preds = sev_preds.detach().cpu().numpy()
            sev_labels = b_sev_labels.to('cpu').numpy()

            st_preds = st_preds.detach().cpu().numpy()
            st_labels = b_st_labels.to('cpu').numpy()

            rebut_preds = rebut_preds.detach().cpu().numpy()
            rebut_labels = b_rebut_labels.to('cpu').numpy()

            return total_loss, model, \
                   sev_preds, sev_labels, \
                    st_preds, st_labels, \
                    rebut_preds, rebut_labels


def retrieve_preds(batch, device, model):
    # tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_user_infos = batch[2].to(device)
    b_sev_labels = batch[3].to(device)
    b_st_labels = batch[4].to(device)
    b_rebut_labels = batch[5].to(device)
    b_topic_labels = batch[6].to(device)

    sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids,
                                                         token_type_ids=None,
                                                         attention_mask=b_input_mask,
                                                         user_infos=b_user_infos,
                                                         sev_labels=b_sev_labels,
                                                         st_labels=b_st_labels,
                                                         rebut_labels=b_rebut_labels)

    return sev_preds, st_preds, rebut_preds, total_loss, model

def inner_zs(batch, device, model, type="train"):
    b_a_input_ids = batch[0].to(device)
    b_p_input_ids = batch[1].to(device)
    b_n_input_ids = batch[2].to(device)

    b_a_mask = batch[3].to(device)
    b_p_mask = batch[4].to(device)
    b_n_mask = batch[5].to(device)

    b_user_infos = batch[6].to(device)
    b_sev_labels = batch[7].to(device)
    b_st_labels = batch[8].to(device)
    b_rebut_labels = batch[9].to(device)
    b_topic_labels = batch[10].to(device)

    if type=='train':
        model.zero_grad()
        sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_zs(batch, device, model)
        return total_loss, model

    elif type=='val':
        with torch.no_grad():
            sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_zs(batch, device, model)

            # Severity test scikit
            sev_preds = sev_preds.detach().cpu().numpy()
            sev_labels = b_sev_labels.to('cpu').numpy()

            st_preds = st_preds.detach().cpu().numpy()
            st_labels = b_st_labels.to('cpu').numpy()

            rebut_preds = rebut_preds.detach().cpu().numpy()
            rebut_labels = b_rebut_labels.to('cpu').numpy()

            return total_loss, model, \
                   sev_preds, sev_labels, \
                    st_preds, st_labels, \
                    rebut_preds, rebut_labels

def retrieve_preds_zs(batch, device, model):
    # tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs
    b_a_input_ids = batch[0].to(device)
    b_p_input_ids = batch[1].to(device)
    b_n_input_ids = batch[2].to(device)

    b_a_mask = batch[3].to(device)
    b_p_mask = batch[4].to(device)
    b_n_mask = batch[5].to(device)

    b_user_infos = batch[6].to(device)
    b_sev_labels = batch[7].to(device)
    b_st_labels = batch[8].to(device)
    b_rebut_labels = batch[9].to(device)
    b_topic_labels = batch[10].to(device)

    sev_preds, st_preds, rebut_preds, total_loss = model(b_a_input_ids, b_p_input_ids, b_n_input_ids,
                                                         b_a_mask, b_p_mask, b_n_mask,
                                                         user_infos=b_user_infos,
                                                         sev_labels=b_sev_labels,
                                                         st_labels=b_st_labels,
                                                         rebut_labels=b_rebut_labels)

    return sev_preds, st_preds, rebut_preds, total_loss, model


def inner_zs_online(batch, device, model, complete_data_dict, topics_dict, type="train", ):
    b_data = zsu.sample_topics_online(batch, complete_data_dict, topics_dict)


    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)

    batch_size = len(batch[0])
    #print(batch)
    if type == 'train':
        model.zero_grad()
        sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_zs_online(b_data, device, model, batch_size)
        return total_loss, model

    elif type=='val':
        with torch.no_grad():
            sev_preds, st_preds, rebut_preds, total_loss, model = retrieve_preds_zs_online(b_data, device, model, batch_size)

            # Severity test scikit
            sev_preds = sev_preds.detach().cpu().numpy()
            sev_labels = b_sev_labels.to('cpu').numpy()

            st_preds = st_preds.detach().cpu().numpy()
            st_labels = b_st_labels.to('cpu').numpy()

            rebut_preds = rebut_preds.detach().cpu().numpy()
            rebut_labels = b_rebut_labels.to('cpu').numpy()

            return total_loss, model, \
                   sev_preds, sev_labels, \
                    st_preds, st_labels, \
                    rebut_preds, rebut_labels

def retrieve_preds_zs_online(b_data, device, model, batch_size):
    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)

    sev_preds, st_preds, rebut_preds, total_loss = model(b_a_input_ids, b_p_input_ids, b_n_input_ids,
                                                         b_a_mask, b_p_mask, b_n_mask,
                                                         user_infos=b_user_infos,
                                                         sev_labels=b_sev_labels,
                                                         st_labels=b_st_labels,
                                                         rebut_labels=b_rebut_labels,
                                                         batch_size=batch_size)

    return sev_preds, st_preds, rebut_preds, total_loss, model

def inner_top_embeddings(batch, device, model, complete_data_dict, topics_dict, type="train", ):
    b_data = zsu.sample_topics_online(batch, complete_data_dict, topics_dict)


    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)

    batch_size = len(batch[0])
    #print(batch)
    if type == 'train':
        model.zero_grad()
        topic_embeddings, model = retrieve_top_embeddings(b_data, device, model, batch_size)
        return topic_embeddings, model

    elif type=='val':
        with torch.no_grad():
            topic_embeddings, model = retrieve_top_embeddings(b_data, device, model, batch_size)

        return topic_embeddings, model


def retrieve_top_embeddings(b_data, device, model, batch_size):
    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)


    topic_embeddings = model(b_a_input_ids, None, None,
                             b_a_mask, None, None,
                             user_infos=b_user_infos,
                             sev_labels=b_sev_labels,
                             st_labels=b_st_labels,
                             rebut_labels=b_rebut_labels,
                             batch_size=batch_size)

    return topic_embeddings, model

def inner_sestre(batch, device, model, complete_data_dict, topics_dict, topic_embeddings, type="train"):
    b_data = zsu.sample_topics_online(batch, complete_data_dict, topics_dict)

    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)

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

            return total_loss, model, \
                   sev_preds, sev_labels, \
                   st_preds, st_labels, \
                   rebut_preds, rebut_labels

def retrieve_preds_sestre(b_data, device, model, batch_size, topic_embeddings):
    b_a_input_ids = b_data[0].to(device)
    b_p_input_ids = b_data[1].to(device)
    b_n_input_ids = b_data[2].to(device)

    b_a_mask = b_data[3].to(device)
    b_p_mask = b_data[4].to(device)
    b_n_mask = b_data[5].to(device)

    b_user_infos = b_data[6].to(device)
    b_sev_labels = b_data[7].to(device)
    b_st_labels = b_data[8].to(device)
    b_rebut_labels = b_data[9].to(device)
    b_topic_labels = b_data[10].to(device)

    #print(topic_embeddings)
    sev_preds, st_preds, rebut_preds, total_loss = model(b_a_input_ids, token_type_ids=None, attention_mask=None,
                                                         topic_embedding=topic_embeddings,
                                                         user_infos=b_user_infos,
                                                         sev_labels=b_sev_labels, st_labels=b_st_labels, rebut_labels=b_rebut_labels,
                                                         topic_labels=b_topic_labels,
                                                         batch_size=4)

    return sev_preds, st_preds, rebut_preds, total_loss, model