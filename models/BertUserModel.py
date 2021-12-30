import torch
from transformers import BertModel, BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

##############################################
###  BERT FOR SINGLELABEL CLASSIFICATION   ###
##############################################
class BertUserModel(BertPreTrainedModel):
    def __init__(self, config, num_user_info=7, num_severity=4, num_stances=3, num_rebuttals=2):
        super(BertUserModel, self).__init__(config)

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

        # Fully connected layer for severity classification: Real, Not Severe, Possibly Severe, Highly Severe
        self.severity_part = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)
        #self.severity_part = torch.nn.Linear(in_features=(config.hidden_size + self.num_user_info), out_features=100)
        self.severity_classifier = torch.nn.Linear(in_features=100, out_features=num_severity)

        # Fully connected layer for stance classification: Support, Deny, Neither
        self.stance_part = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)
        #self.stance_part = torch.nn.Linear(in_features=(config.hidden_size + self.num_user_info), out_features=100)
        self.stance_classifier = torch.nn.Linear(in_features=100, out_features=num_stances)

        # Fully connected layer for rebuttal classification: True, False
        self.rebuttal_part = torch.nn.Linear(in_features=(config.hidden_size), out_features=100)
        #self.rebuttal_part = torch.nn.Linear(in_features=(config.hidden_size + self.num_user_info), out_features=100)
        self.rebuttal_classifier = torch.nn.Linear(in_features=100, out_features=num_rebuttals)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, user_infos=None, sev_labels=None, st_labels=None, rebut_labels=None):
        bert_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = bert_output[1]

        pooled_output = self.dropout(pooled_output)


        #pooleduser = torch.cat((pooled_output, user_infos), dim=1)

        hidden_sev = self.severity_part(pooled_output)
        #hidden_sev = self.severity_part(pooleduser)
        sev_logits = self.severity_classifier(hidden_sev)
        #sev_preds = self.softmax(sev_logits)

        hidden_st = self.stance_part(pooled_output)
        #hidden_st = self.stance_part(pooleduser)
        st_logits = self.stance_classifier(hidden_st)
        #st_preds = self.softmax(st_logits)

        hidden_rebut = self.rebuttal_part(pooled_output)
        #hidden_rebut = self.rebuttal_part(pooleduser)
        rebut_logits = self.rebuttal_classifier(hidden_rebut)
        #rebut_preds = self.softmax(rebut_logits)

        loss_fct = CrossEntropyLoss()

        # Assuming that if sev_labels are present, all labels are present
        if sev_labels is not None:
            sev_loss = loss_fct(sev_logits, sev_labels)
            st_loss = loss_fct(st_logits, st_labels)
            rebut_loss = loss_fct(rebut_logits, rebut_labels)

            total_loss = sev_loss + st_loss + rebut_loss

            # This will return the indices instead of the values: _ in place of values
            _, sev_preds = torch.max(sev_logits, 1)
            _, st_preds = torch.max(st_logits, 1)
            _, rebut_preds = torch.max(rebut_logits, 1)

            total_loss = total_loss.mean()
            return sev_preds, st_preds, rebut_preds, total_loss
            #return sev_preds, st_preds, rebut_preds, sev_loss, st_loss, rebut_loss
        else:
            # This will return the indices instead of the values: _ in place of values
            _, sev_preds = torch.max(sev_logits, 1)
            _, st_preds = torch.max(st_logits, 1)
            _, rebut_preds = torch.max(rebut_logits, 1)
            return sev_preds, st_preds, rebut_preds



