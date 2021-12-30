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
import models.BertUserModel as bum
import random

logging.basicConfig(level=logging.ERROR)

severity_list = ['REAL', 'NOT_SEVERE', 'POSSIBLY_SEVERE', 'HIGHLY_SEVERE']

### SEEDIFY ME ###
seed_val = 42 # answer of the universe
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


### USE GPU IF APPLICABLE ###
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### CONFIG ###
#tweet_file = 'data/unmerged/annod_data.json'
tweet_file = 'data/unmerged/all_merged_data.json'
#tweet_graph_file = 'tweet_graph.json'
#tweet_annotation_file = 'data/labeled_misinformation_real.json'
# YEET
tweet_annotation_file = 'data/labeled_misinformation_real_dupeless.json'
#input_model_dir = '/shared/hltdir4/disk1/team/data/models/bert/emot-det-covid-2/'
output_dir = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/w-user-info'
output_dir_tokenizer = '/shared/hltdir4/disk1/team/data/models/bert/oa-cov/tokenizer/'

#-------------------------------#
print("Creating dict, graph, labels. ")
#-------------------------------#

# tweet ids with all of their information
#tweet_dict = du.create_tweet_dict(tweet_file)
tweet_dict = du.load_tweet_dict(tweet_file)

# tweet graph with all of the edges between nodes
tweet_graph = du.create_twitter_graph(tweet_dict) # tweet graph in the form of tweet, context (tweet is tweet, context is tweet it replies to, null if none)

# tweet annotation: dict[id][LABEL] where LABEL is [severity][stance][rebuttal]
tweet_anno_dict = du.create_tweet_label_dict(tweet_annotation_file)

### GROUP THE DATA WE ACTUALLY WANT TOGETHER ###
# First thing, take all of the annotated files and find the values we want
# Sentences, User information (as list), labels
sentences, user_infos, sev_labs, st_labs, rebut_labs = du.retrieve_anno_ids(tweet_anno_dict, tweet_dict)
length = len(sentences)
tr = int(.7 * length)
val = int(.1 * length)

# Shuffle
d = list(zip(sentences, user_infos, sev_labs, st_labs, rebut_labs))
random.shuffle(d)
sentences, user_infos, sev_labs, st_labs, rebut_labs = zip(*d)


### TOKENIZATION GOODNESS ###
#-------------------------------#
print("Tokenizing. ")
#-------------------------------#
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 200  #PLACEHOLDER

# training stuffs #
tr_sentences, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs = sentences[:tr], user_infos[:tr], \
                                                                            sev_labs[:tr], st_labs[:tr], rebut_labs[:tr]

# valid stuffs #
val_sentences, val_user_infos, val_sev_labs, val_st_labs, val_rebut_labs = sentences[tr:], user_infos[tr:], \
                                                                                sev_labs[tr:], st_labs[tr:], rebut_labs[tr:]

# TOKEN IDS #
#-------------------------------#
print("Tensifying. ")
#-------------------------------#
tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs = du.retrieve_ids_and_tensify(tr_sentences,
                                                                                                                      tr_user_infos,
                                                                                                                      tr_sev_labs,
                                                                                                                      tr_st_labs,
                                                                                                                      tr_rebut_labs,
                                                                                                                      tokenizer,
                                                                                                                      maximum_length=max_len)
val_input_ids, val_attention_masks, val_user_infos, val_sev_labs, val_st_labs, val_rebut_labs = du.retrieve_ids_and_tensify(val_sentences,
                                                                                                                            val_user_infos,
                                                                                                                            val_sev_labs,
                                                                                                                            val_st_labs,
                                                                                                                            val_rebut_labs,
                                                                                                                            tokenizer,
                                                                                                                            maximum_length=max_len
                                                                                                                            )

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 4
train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_user_infos, val_sev_labs, val_st_labs, val_rebut_labs)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              #sampler = RandomSampler(train_dataset),
                              batch_size = batch_size)

validation_dataloader = DataLoader(val_dataset,
                                   #sampler = SequentialSampler(val_dataset),
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

model = bum.BertUserModel.from_pretrained(
    'bert-base-uncased',
    num_user_info=len(user_infos[0]),
    num_severity=4,
    num_stances=3,
    num_rebuttals=2,
    output_attentions = False,
    output_hidden_states = False
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
            if step % 10 == 0 and not step == 0:
                elapsed = du.format_time(time.time() - t0)
                print(f'Batch{step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}')

            # tr_input_ids, tr_attention_masks, tr_user_infos, tr_sev_labs, tr_st_labs, tr_rebut_labs
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_user_infos = batch[2].to(device)
            b_sev_labels = batch[3].to(device)
            b_st_labels = batch[4].to(device)
            b_rebut_labels = batch[5].to(device)

            ### ROUND 1 - Not using user_infos ###
            model.zero_grad()

            ### SEPARATE LOSS ###
            # sev_preds, st_preds, rebut_preds, sev_loss, st_loss, rebut_loss = model(b_input_ids,
            #                                                                           token_type_ids=None,
            #                                                                           attention_mask=b_input_mask,
            #                                                                           sev_labels=b_sev_labels,
            #                                                                           st_labels=b_st_labels,
            #                                                                           rebut_labels=b_rebut_labels)

            ### COMBINED LOSS ###
            # sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids,
            #                                                                         token_type_ids=None,
            #                                                                         attention_mask=b_input_mask,
            #                                                                         sev_labels=b_sev_labels,
            #                                                                         st_labels=b_st_labels,
            #                                                                         rebut_labels=b_rebut_labels)

            sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids,
                                                                 token_type_ids=None,
                                                                 attention_mask=b_input_mask,
                                                                 user_infos=b_user_infos,
                                                                 sev_labels=b_sev_labels,
                                                                 st_labels=b_st_labels,
                                                                 rebut_labels=b_rebut_labels)
            #print(f'LABELS: {b_sev_labels}, {b_st_labels}, {b_rebut_labels}')
            #print(f'PREDS: {sev_preds}, {st_preds}, {rebut_preds}')
            # Get the losses
            # total_train_loss += sev_loss.item()
            # total_train_loss += st_loss.item()
            # total_train_loss += rebut_loss.item()
            total_train_loss += total_loss.item()

            # sev_loss.backward(retain_graph=True)
            # st_loss.backward(retain_graph=True)
            # rebut_loss.backward()
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

        # Severity: acc, tp, fp, tn, fn
        total_sev_acc = 0
        total_sev_TP = 0
        total_sev_FP = 0
        total_sev_TN = 0
        total_sev_FN = 0

        # Stance: acc, tp, fp, tn, fn
        total_st_acc = 0
        total_st_TP = 0
        total_st_FP = 0
        total_st_TN = 0
        total_st_FN = 0

        # Rebuttal: acc, tp, fp, tn, fn
        total_reb_acc = 0
        total_reb_TP = 0
        total_reb_FP = 0
        total_reb_TN = 0
        total_reb_FN = 0



        sev_preds_total = []
        sev_labels_total = []

        st_preds_total = []
        st_labels_total = []

        rebut_preds_total = []
        rebut_labels_total = []
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_user_infos = batch[2].to(device)
            b_sev_labels = batch[3].to(device)
            b_st_labels = batch[4].to(device)
            b_rebut_labels = batch[5].to(device)

            with torch.no_grad():
                sev_preds, st_preds, rebut_preds, total_loss = model(b_input_ids,
                                                                                        token_type_ids=None,
                                                                                        attention_mask=b_input_mask,
                                                                                         user_infos=b_user_infos,
                                                                                        sev_labels=b_sev_labels,
                                                                                        st_labels=b_st_labels,
                                                                                        rebut_labels=b_rebut_labels)

                # Get the losses
                # total_eval_loss += sev_loss.item()
                # total_eval_loss += st_loss.item()
                # total_eval_loss += rebut_loss.item()
                total_eval_loss += total_loss.item()

                # Severity test scikit
                sev_preds = sev_preds.detach().cpu().numpy()
                sev_labels = b_sev_labels.to('cpu').numpy()
                sev_preds_total.append(sev_preds)
                sev_labels_total.append(sev_labels)

                # Stance test scikit
                st_preds = st_preds.detach().cpu().numpy()
                st_labels = b_st_labels.to('cpu').numpy()
                st_preds_total.append(st_preds)
                st_labels_total.append(st_labels)

                # Rebuttal test scikit
                rebut_preds = rebut_preds.detach().cpu().numpy()
                rebut_labels = b_rebut_labels.to('cpu').numpy()
                rebut_preds_total.append(rebut_preds)
                rebut_labels_total.append(rebut_labels)

                # # Severity metrics
                # #sev_preds = sev_preds.detach().cpu().numpy()
                # #sev_labels = b_sev_labels.to('cpu').numpy()
                # b_sev_acc, b_sev_TP, b_sev_FP, b_sev_TN, b_sev_FN = du.apn(sev_labels, sev_preds, list_type='severity')
                # total_sev_acc += b_sev_acc
                # total_sev_TP, total_sev_FP, total_sev_TN, total_sev_FN = du.sum_pn(total_sev_TP, total_sev_FP, total_sev_TN, total_sev_FN,
                #                                                                    b_sev_TP, b_sev_FP, b_sev_TN, b_sev_FN)
                #
                #
                #
                # # Stance metrics
                # #st_preds = st_preds.detach().cpu().numpy()
                # #st_labels = b_st_labels.to('cpu').numpy()
                # b_st_acc, b_st_TP, b_st_FP, b_st_TN, b_st_FN = du.apn(st_labels, st_preds, list_type='stance')
                # total_st_acc += b_st_acc
                # total_st_TP, total_st_FP, total_st_TN, total_st_FN = du.sum_pn(total_st_TP, total_st_FP, total_st_TN, total_st_FN,
                #                                                                b_st_TP, b_st_FP, b_st_TN, b_st_FN)
                #
                #
                # # Rebuttal metrics
                # rebut_preds = rebut_preds.detach().cpu().numpy()
                # rebut_labels = b_rebut_labels.to('cpu').numpy()
                # b_rebut_acc, b_rebut_TP, b_rebut_FP, b_rebut_TN, b_rebut_FN = du.apn(rebut_labels, rebut_preds, list_type='rebuttal')
                # total_reb_acc += b_rebut_acc
                # total_reb_TP, total_reb_FP, total_reb_TN, total_reb_FN = du.sum_pn(total_reb_TP, total_reb_FP, total_reb_TN, total_reb_FN,
                #                                                                    b_rebut_TP, b_rebut_FP, b_rebut_TN, b_rebut_FN)

        sev_preds_total = np.concatenate(sev_preds_total)
        sev_labels_total = np.concatenate(sev_labels_total)

        st_preds_total = np.concatenate(st_preds_total)
        st_labels_total = np.concatenate(st_labels_total)

        rebut_preds_total = np.concatenate(rebut_preds_total)
        rebut_labels_total = np.concatenate(rebut_labels_total)

        du.apn_skl(sev_labels_total, sev_preds_total, 'Severity')
        du.apn_skl(st_preds_total, st_labels_total, 'Stance')
        du.apn_skl(rebut_preds_total, rebut_labels_total, 'Rebuttal')
        print('='*30)
        validation_time = format(time.time() - t0)

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        avg_sev_acc = total_sev_acc / len(validation_dataloader)
        avg_st_acc = total_st_acc / len(validation_dataloader)
        avg_reb_acc = total_reb_acc / len(validation_dataloader)

        # print(f'---------- Severity Scores ----------')
        # du.pre_rec_f1(acc=avg_sev_acc,
        #                    tp=total_sev_TP,
        #                    fp=total_sev_FP,
        #                    tn=total_sev_TN,
        #                    fn=total_sev_FN,
        #                    type='severity')
        #
        # print(f'---------- Stance Scores ----------')
        # du.pre_rec_f1(acc=avg_st_acc,
        #               tp=total_st_TP,
        #               fp=total_st_FP,
        #               tn=total_st_TN,
        #               fn=total_st_FN,
        #               type='stance')
        #
        # print(f'---------- Rebuttal Scores ----------')
        # du.pre_rec_f1(acc=avg_reb_acc,
        #               tp=total_reb_TP,
        #               fp=total_reb_FP,
        #               tn=total_reb_TN,
        #               fn=total_reb_FN,
        #               type='rebuttal')

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        last_validation_loss = avg_val_loss
        last_updated_epoch = epoch_i
        # torch.save({'epoch': epoch_i,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': avg_val_loss,
        #             }, output_dir + 'ckpt/ckpt-' + str(epoch_i) + '.pt')

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

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir_tokenizer)
