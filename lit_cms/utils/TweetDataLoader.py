import torch
from torch.utils.data import Dataset
from lit_cms.utils.LitUtils import retrieve_data

class TweetDataSet(Dataset):
    def __init__(self, ids, complete_data_dict, topic_dict):
        b_data = retrieve_data(ids, complete_data_dict, topic_dict)

        self.input_ids = b_data["input_ids"]
        self.masks = b_data["masks"]
        self.user_infos = b_data["user_infos"]
        self.severity_labels = b_data["severity_labels"]
        self.stance_labels = b_data["stance_labels"]
        self.rebuttal_labels = b_data["rebuttal_labels"]
        self.topic_labels = b_data["topic_labels"]
        self.theme_labels = b_data["theme_labels"]
        self.tweet_ids = b_data["tweet_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tweet_id = self.tweet_ids[idx]
        input_ids = self.input_ids[idx]
        mask = self.masks[idx]
        user_info = self.user_infos[idx]
        severity_label = self.severity_labels[idx]
        stance_label = self.stance_labels[idx]
        rebuttal_label = self.rebuttal_labels[idx]
        topic_label = self.topic_labels[idx]
        theme_label = self.theme_labels[idx]

        item = {"input_ids": input_ids,
                  "mask": mask,
                  "user_info": user_info,
                  "severity_label": severity_label,
                  "stance_label": stance_label,
                  "rebuttal_label": rebuttal_label,
                  "topic_label": topic_label,
                  "theme_label": theme_label,
                  "tweet_id": tweet_id}

        return item