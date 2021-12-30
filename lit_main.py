import torch
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import time
import utils.data_utils as du
import utils.train_test_utils as ttu
import utils.zero_shot_utils as zsu
import utils.zero_shot_inference as zsi
import utils.sestre_utils as ses
from pytorch_lightning.loggers import NeptuneLogger
import os
import logging

import random

from pytorch_lightning import Trainer, seed_everything
from lit_cms.models.LitBERT import LitBERT
from lit_cms.config import Config
from lit_cms.config import GetModel
from lit_cms.utils.LitUtils import retrieve_data
from lit_cms.utils.TweetDataLoader import TweetDataSet
if __name__ == '__main__':
    config = Config()

    #seed_everything(config.args.seed)
    random.seed(config.args.seed)
    np.random.seed(config.args.seed)
    torch.manual_seed(config.args.seed)

    logging.basicConfig(level=logging.ERROR)


    tweet_dict = du.load_tweet_dict(config.args.merged_tweet_file)
    tweet_annotations = du.create_tweet_label_dict(config.args.tweet_annotation_file)

    training_ids, training_topics, validation_ids, validation_topics, \
        test_ids, test_topics, complete_data_dict = zsu.parse_training_validation_test(tweet_annotations, tweet_dict, config.args.seed)


    train_dataset = TweetDataSet(training_ids, complete_data_dict, training_topics)
    val_dataset = TweetDataSet(validation_ids, complete_data_dict, validation_topics)


    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size = config.args.batch_size,
                                  pin_memory = True)

    validation_dataloader = DataLoader(val_dataset,
                                       batch_size = config.args.batch_size,
                                       pin_memory = True)


    classification_model = GetModel(config, complete_data_dict, training_topics, validation_topics, test_topics)

    PARAMS = {'max_epochs': config.args.num_epochs,
              'learning_rate': config.args.learning_rate,
              'batch_size': config.args.batch_size}

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name=config.args.project_name,
        experiment_name=config.args.experiment_name,
        params={"max_epochs": config.args.num_epochs},
    )


    trainer = Trainer(gpus=2,
                      max_epochs=config.args.num_epochs,
                      logger=neptune_logger,
                      gradient_clip_val=1,
                      )
    trainer.fit(classification_model, train_dataloader, validation_dataloader)

