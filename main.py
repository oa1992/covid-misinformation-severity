'''
Step 1: Base Program
Take a tweet and it's context (the tweet it is replying to):
    - Evaluate its severity (Real, Not Severe, Possibly Severe, Highly Severe)
        . This form shall be a tuple (tweet, context)

'''

import torch
import numpy as np
import time
import logging
import utils.data_utils as du
import utils.train_test_utils as ttu
import utils.zero_shot_utils as zsu
#import models.BertUserModel as bum
import models.ZeroShotSeverityModel as zsm
import random


logging.basicConfig(level=logging.ERROR)

severity_list = ['REAL', 'NOT_SEVERE', 'POSSIBLY_SEVERE', 'HIGHLY_SEVERE']

### SEEDIFY ME ###
seed_val = 79 # answer of the universe
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


### USE GPU IF APPLICABLE ###
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### CONFIG ###
# tweet_file = 'data/unmerged/annod_data.json'
tweet_file = 'data/unmerged/all_merged_data.json'
# tweet_graph_file = 'tweet_graph.json'
# tweet_annotation_file = 'data/labeled_misinformation_real.json'
# YEET
tweet_annotation_file = 'data/last_pull.json'
# input_model_dir = '/shared/hltdir4/disk1/team/data/models/bert/emot-det-covid-2/'
output_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/zero-shot/zs-79/'
output_dir_tokenizer = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/tokenizer/'

#-------------------------------#
print("Creating dict, graph, labels. ")
#-------------------------------#

# tweet ids with all of their information
# tweet_dict = du.create_tweet_dict(tweet_file)
tweet_dict = du.load_tweet_dict(tweet_file)

# tweet graph with all of the edges between nodes
tweet_graph = du.create_twitter_graph(tweet_dict) # tweet graph in the form of tweet, context (tweet is tweet, context is tweet it replies to, null if none)

# tweet annotation: dict[id][LABEL] where LABEL is [severity][stance][rebuttal][topic]
tweet_anno_dict = du.create_tweet_label_dict(tweet_annotation_file)


# training_data, validation_data = du.parse_and_tensify(tweet_anno_dict, tweet_dict)
#
#
# tr_a_input_ids, tr_p_input_ids, tr_n_input_ids, tr_a_attention_masks, tr_p_attention_masks, tr_n_attention_masks,\
#     tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs, tr_topic_labs = training_data
# val_a_input_ids, val_p_input_ids, val_n_input_ids, val_a_attention_masks, val_p_attention_masks, val_n_attention_masks,\
#     val_user_infos, val_sev_labs, val_st_labs, val_rebut_labs, val_topic_labs = validation_data

training_ids, training_topics, validation_ids, validation_topics, \
    testing_ids, testing_topics, complete_data_dict = zsu.parse_training_validation_test(tweet_anno_dict, tweet_dict, seed_val)

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

print(f"lengths {len(training_ids)} - {len(validation_ids)} - {len(testing_ids)}")

batch_size = 4
# train_dataset = TensorDataset(tr_a_input_ids, tr_p_input_ids, tr_n_input_ids,
#                               tr_a_attention_masks, tr_p_attention_masks, tr_n_attention_masks,
#                               tr_user_infos,
#                               tr_sev_labs,
#                               tr_st_labs,
#                               tr_rebut_labs,
#                               tr_topic_labs)
#
# val_dataset = TensorDataset(val_a_input_ids, val_p_input_ids, val_n_input_ids,
#                             val_a_attention_masks, val_p_attention_masks, val_n_attention_masks,
#                             val_user_infos,
#                             val_sev_labs,
#                             val_st_labs,
#                             val_rebut_labs,
#                             val_topic_labs)

train_dataset = TensorDataset(training_ids)
val_dataset = TensorDataset(validation_ids)
test_dataset = TensorDataset(testing_ids)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              #sampler = RandomSampler(train_dataset),
                              batch_size = batch_size)

validation_dataloader = DataLoader(val_dataset,
                                   #sampler = SequentialSampler(val_dataset),
                                   batch_size = batch_size)

test_dataset = DataLoader(test_dataset,
                          batch_size = batch_size)
#-------------------------------#
print("Declaring Model. ")
#-------------------------------#
# model = bum.BertUserModel.from_pretrained(
#     'bert-base-uncased',
#     num_severity=4,
#     num_stances=3,
#     num_rebuttals=2,
#     output_attentions = False,
#     output_hidden_states = False
# )

# model = bum.BertUserModel.from_pretrained(
#     'bert-base-uncased',
#     num_user_info=len(tr_user_infos[0]),
#     num_severity=4,
#     num_stances=3,
#     num_rebuttals=2,
#     output_attentions = False,
#     output_hidden_states = False
# )

model = zsm.ZeroShotSeverityModel.from_pretrained(
    'bert-base-uncased',
    num_user_info=7, #yolo
    num_severity=4,
    num_stances=3,
    num_rebuttals=2,
    output_attentions=False,
    output_hidden_states=False
)

model.cuda()

from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)

from transformers import get_linear_schedule_with_warmup

epochs = 20
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


training_stats = []

total_t0 = time.time()

last_validation_loss = 99999999999999999
last_updated_epoch = 0
EARLY_STOP = False
for epoch_i in range(0, epochs):
    if not EARLY_STOP:
        print("")
        print(f'============ Epoch {epoch_i + 1} / {epochs} ============')
        print('Training...')

        t0 = time.time()

        # Reset the total loss for this epoch
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 10 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = du.format_time(time.time() - t0)
                print(f'Batch{step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}')


            total_loss, model = ttu.inner_zs_online(batch, device, model,
                                                    complete_data_dict, training_topics, type='train')

            total_train_loss += total_loss.item()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = du.format_time(time.time() - t0)

        print("")
        print(f" Average training loss: {avg_train_loss:.2f}")
        print(f" Training epoch took: {training_time}")

        print("")
        print("Running Validation...")

        t0 = time.time()
        # Put the model in evaluation mode -- the dropout layers behave differently during evaluation
        model.eval()

        ### Tracking variables for metrics ###
        total_eval_loss = 0
        nb_eval_steps = 0

        sev_preds_total = []
        sev_labels_total = []

        st_preds_total = []
        st_labels_total = []

        rebut_preds_total = []
        rebut_labels_total = []
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            total_loss, model, sev_preds, sev_labels, \
            st_preds, st_labels, rebut_preds, rebut_labels = ttu.inner_zs_online(batch, device, model,
                                                                          complete_data_dict, validation_topics, type='val')

            # Update Loss
            total_eval_loss += total_loss.item()

            # Add predictions and labels to their respective arrays
            sev_preds_total.append(sev_preds)
            sev_labels_total.append(sev_labels)

            st_preds_total.append(st_preds)
            st_labels_total.append(st_labels)

            rebut_preds_total.append(rebut_preds)
            rebut_labels_total.append(rebut_labels)

        ################################################################
        ###               Concat Array - Batch to One                ###
        ###                                                          ###
        sev_preds_total = np.concatenate(sev_preds_total)
        sev_labels_total = np.concatenate(sev_labels_total)

        st_preds_total = np.concatenate(st_preds_total)
        st_labels_total = np.concatenate(st_labels_total)

        rebut_preds_total = np.concatenate(rebut_preds_total)
        rebut_labels_total = np.concatenate(rebut_labels_total)
        ###                                                          ###
        ###                    END CONCAT OF ARRAY                   ###
        ################################################################

        # du.apn_skl(sev_labels_total, sev_preds_total, 'Severity')
        # du.apn_skl(st_preds_total, st_labels_total, 'Stance')
        # du.apn_skl(rebut_preds_total, rebut_labels_total, 'Rebuttal')
        print('='*30)
        validation_time = format(time.time() - t0)

        avg_val_loss = total_eval_loss / len(validation_dataloader)


        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        if last_validation_loss > avg_val_loss:
            last_validation_loss = avg_val_loss
            last_updated_epoch = epoch_i
            try:
                os.mkdir(output_dir + 'ckpt/')
            except:
                pass
            torch.save({'epoch': epoch_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                        }, output_dir + 'ckpt/ckpt-' + str(epoch_i) + '.pt')

        if (epoch_i - last_updated_epoch) >= 4:
            EARLY_STOP = True

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                #'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    else:
        break

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(du.format_time(time.time() - total_t0)))

    import pandas as pd

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)

    ######################################################################
    ### SAVING AND LOADING FINE-TUNED MODEL ###
    import os

    # Saving best-practices: if you use defaults names for the model you can reload it using from_pretrained()

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    justin = 0
    if last_validation_loss > avg_val_loss:
        last_validation_loss = avg_val_loss

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        justin = 0
    else:
        justin += 1

    if justin >= 5:
        printf("Quitting Early...")
        break
    # tokenizer.save_pretrained(output_dir_tokenizer)
