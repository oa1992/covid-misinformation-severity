import torch
from transformers import BertModel, BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class ZeroShotSeverityModel(BertPreTrainedModel):
    def __init__(self, config, num_user_info=7, num_severity=4, num_stances=3, num_rebuttals=2):
        super(ZeroShotSeverityModel, self).__init__(config)

        self.num_user_info = num_user_info

        # Actual number of severity classes
        self.num_severity = num_severity

        # Actual number of stances
        self.num_stance = num_stances

        # Actual number of rebuttals
        self.num_rebuttals = num_rebuttals

        # Bert must be passed its config
        self.bert = BertModel(config)

        # Dropout
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Reduce the features for tweet, positive example, negative example
        self.tweet_reduce = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)
        #self.pos_reduce = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)
        #self.neg_reduce = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)

        # Classifiers for each thingy
        self.severity_classifier = torch.nn.Linear(in_features=100, out_features=num_severity)
        self.stance_classifier = torch.nn.Linear(in_features=(config.hidden_size), out_features=num_stances)
        self.rebuttal_classifier = torch.nn.Linear(in_features=(config.hidden_size), out_features=num_rebuttals)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.init_weights()

    def forward(self, anchor_iid=None, positive_iid=None, negative_iid=None,
                      anchor_mask=None, positive_mask=None, negative_mask=None,
                      anchor_ttid=None, positive_ttid=None, negative_ttid=None,
                      user_infos=None, sev_labels=None, st_labels=None, rebut_labels=None, topic_labels=None,
                      batch_size=4):
              #       sev_labels=b_sev_labels,
              #       st_labels=b_st_labels,
              #       rebut_labels=b_rebut_labels)
              #
              #   iid_tweet, ttid_tweet, atnm_tweet, sev_tweet_label,
              # iid_pos, ttid_pos, atnm_pos,
              # iid_neg, ttid_neg, atnm_neg,
              # st_tweet_label, reb_tweet_label, topic_tweet_label, batch_size):
        ############################################################################
        ### Create the embeddings for the tweet, positive, and negative examples ###
        ############################################################################
        # Step 1: concatenate each iid, ttid, and atnm on the batch layer, then redistribute.
        # ex: iid_tweet is [4, whatever]
        # results of bert will be [4, whatever]
        # concatenate to [12, whatever], then separate bert output
        yolo = False
        if positive_iid != None:
            b_dim = 0
            input_ids = torch.cat((anchor_iid, positive_iid, negative_iid), b_dim)
            token_type_ids = None

            attention_mask = torch.cat((anchor_mask, positive_mask, negative_mask), b_dim)

            bert_output = self.bert(input_ids, token_type_ids, attention_mask)

            total_pooled_output = bert_output[1]
            total_pooled_output = self.dropout(total_pooled_output)

            anchor_output = total_pooled_output[:batch_size]
            reduced_output = self.tweet_reduce(total_pooled_output)

            reduced_tweet = reduced_output[:batch_size]
            reduced_pos = reduced_output[batch_size:batch_size*2]
            reduced_neg = reduced_output[batch_size*2:]

            ############################################################################
            ###       Reduce the size of the embedding to a smaller dimension        ###
            ############################################################################
            # reduced_tweet =  self.tweet_reduce(pooled_tweet)
            # reduced_pos   =  self.pos_reduce(pooled_pos)
            # reduced_neg   =  self.neg_reduce(pooled_neg)

            triplet_loss_func = torch.nn.TripletMarginLoss(margin=1.0)
            ce_loss_fct = CrossEntropyLoss()

            triplet_loss = triplet_loss_func(reduced_tweet, reduced_pos, reduced_neg)


            sev_logits = self.severity_classifier(reduced_tweet)
            st_logits = self.stance_classifier(anchor_output)
            rebut_logits = self.rebuttal_classifier(anchor_output)

            if sev_labels is not None:
                sev_loss = ce_loss_fct(sev_logits, sev_labels)
                st_loss = ce_loss_fct(st_logits, st_labels)
                rebut_loss = ce_loss_fct(rebut_logits, rebut_labels)

                #total_loss = sev_loss + st_loss + rebut_loss + triplet_loss
                total_loss = triplet_loss
                # This will return the indices instead of the values: _ in place of values
                _, sev_preds = torch.max(sev_logits, 1)
                _, st_preds = torch.max(st_logits, 1)
                _, rebut_preds = torch.max(rebut_logits, 1)

                total_loss = total_loss.mean()
                return sev_preds, st_preds, rebut_preds, total_loss
                # return sev_preds, st_preds, rebut_preds, sev_loss, st_loss, rebut_loss
            else:
                # This will return the indices instead of the values: _ in place of values
                _, sev_preds = torch.max(sev_logits, 1)
                _, st_preds = torch.max(st_logits, 1)
                _, rebut_preds = torch.max(rebut_logits, 1)
                return reduced_tweet #,sev_preds, st_preds, rebut_preds
        elif yolo:
            bert_output = self.bert(anchor_iid, anchor_ttid, anchor_mask)
            unpooled_output = bert_output[0]
            return unpooled_output
        else:
            bert_output = self.bert(anchor_iid, anchor_ttid, anchor_mask)
            total_pooled_output = bert_output[1]
            total_pooled_output = self.dropout(total_pooled_output)

            reduced_tweet = self.tweet_reduce(total_pooled_output)

            return reduced_tweet