import models.ZeroShotSeverityModel as zsm
import torch

def load_model(input_dir):
    model = zsm.ZeroShotSeverityModel.from_pretrained(
        'bert-base-uncased',
        num_user_info=7,  # yolo
        num_severity=4,
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


def inference(model, batch, data_dict):
    model_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/zero-shot/ckpt/ckpt-9.pt'

    #model = load_model(model_dir)
    #model.cuda()
    device = load_device()
    b_iids, b_masks = retrieve_input(batch, data_dict)
    b_iids = b_iids.to(device); b_masks=b_masks.to(device)
    model.eval()
    topic_embedding = model(anchor_iid=b_iids, anchor_mask=b_masks)
    #print(topic_embedding)
    return topic_embedding