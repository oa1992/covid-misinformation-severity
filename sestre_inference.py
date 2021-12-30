import models.sestre as sestre
import torch
import random
import numpy as np
import utils.data_utils as du
import utils.zero_shot_utils as zsu
import utils.sestre_utils as ses
import utils.zero_shot_inference as zsi
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def prebatch():
    seed_val = 79  # YEET
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    ### USE GPU IF APPLICABLE ###
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ### CONFIG ###
    tweet_file = 'data/unmerged/all_merged_data.json'
    tweet_annotation_file = 'data/last_pull.json'
    #input_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/zero-shot/zs-79/ckpt/ckpt-3.pt'
    #output_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/sestre/sestre-79/'
    #output_dir_tokenizer = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/tokenizer/'

    tweet_dict = du.load_tweet_dict(tweet_file)
    tweet_anno_dict = du.create_tweet_label_dict(tweet_annotation_file)
    training_ids, training_topics, validation_ids, validation_topics, \
        testing_ids, testing_topics, complete_data_dict = zsu.parse_training_validation_test(tweet_anno_dict, tweet_dict, seed_val)

    batch_size = 4
    test_dataset = TensorDataset(testing_ids)
    test_dataloader = DataLoader(test_dataset,
                                batch_size = batch_size)

    return complete_data_dict, testing_topics, test_dataloader

def load_sestre_model(input_dir):
    model = sestre.StanceRebuttal.from_pretrained(
        'bert-base-uncased',
        num_user_info=7,  # yolo
        num_severity=3,
        num_stances=3,
        num_rebuttals=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model_info = torch.load(input_dir)
    model.load_state_dict(model_info['model_state_dict'])

    return model


def load_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def retrieve_input(batch, data_dict):
    b_iids = []
    b_masks = []

    for id in batch[0].tolist():
        iid = data_dict[id]["input_ids"]
        mask = data_dict[id]["attention_mask"]

        b_iids.append(iid)
        b_masks.append(mask)

    # Create them as tensors #
    b_iids = torch.cat(b_iids, dim=0)
    b_masks = torch.cat(b_masks, dim=0)

    return b_iids, b_masks

def inference(zsi_model, sestre_model, device):
    complete_data_dict, test_topics, test_dataloader = prebatch()

    ### Tracking variables for metrics ###
    total_eval_loss = 0
    nb_eval_steps = 0

    sev_preds_total = []
    sev_labels_total = []

    st_preds_total = []
    st_labels_total = []

    rebut_preds_total = []
    rebut_labels_total = []

    for step, batch in enumerate(test_dataloader):
        if step % 40 == 0 and not step == 0:
            print(f'Batch{step:>5,} of {len(test_dataloader):>5,}.')

        topic_embeddings = zsi.inference(zsi_model, batch, complete_data_dict)
        total_loss, model_2, sev_preds, sev_labels, \
        st_preds, st_labels, rebut_preds, rebut_labels = ses.inner_sestre(batch, device, sestre_model,
                                                                          complete_data_dict, test_topics,
                                                                          topic_embeddings, type='val')

        total_eval_loss += total_loss.item()

        sev_preds_total.append(sev_preds)
        sev_labels_total.append(sev_labels)

        st_preds_total.append(st_preds)
        st_labels_total.append(st_labels)

        rebut_preds_total.append(rebut_preds)
        rebut_labels_total.append(rebut_labels)

    sev_preds_total = np.concatenate(sev_preds_total)
    sev_labels_total = np.concatenate(sev_labels_total)

    st_preds_total = np.concatenate(st_preds_total)
    st_labels_total = np.concatenate(st_labels_total)

    rebut_preds_total = np.concatenate(rebut_preds_total)
    rebut_labels_total = np.concatenate(rebut_labels_total)

    du.apn_skl(sev_labels_total, sev_preds_total, 'Severity')
    du.apn_skl(st_labels_total, st_preds_total, 'Stance')
    du.apn_skl(rebut_labels_total, rebut_preds_total, 'Rebuttal')
    print('=' * 30)

def dothething():
    zsi_input_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/zero-shot/zs-79/ckpt/ckpt-3.pt'
    device = load_device()
    zsi_model = zsi.load_model(zsi_input_dir)
    zsi_model.cuda()

    sestre_input_dir = "/shared/hltdir4/disk1/team/data/models/bert/oa-cov/sestre/test6/ckpt/ckpt-10.pt"
    sestre_model = load_sestre_model(sestre_input_dir)
    sestre_model.cuda()

    inference(zsi_model, sestre_model, device)

if __name__ == "__main__":
    dothething()