import os
import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import BertModel, BertPreTrainedModel, AutoModel, AutoConfig
from transformers import BertConfig
from pytorch_lightning import metrics
from lit_cms.utils.LitUtils import retrieve_data
from collections import OrderedDict


from pytorch_lightning.core.lightning import LightningModule

class LitBERT(LightningModule):
    def __init__(self, config, complete_data_dict, training_topics, validation_topics, test_topics):
        super(LitBERT, self).__init__()
        self.total_steps = len(training_topics) * config.args.num_epochs
        self.config = config

        self.ac = BertConfig.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)
        self.shared_bert = BertModel.from_pretrained('bert-base-uncased', config=self.ac)#, output_attentions=False, output_hidden_states=False)

        self.bert_conf = self.shared_bert.config

        self.dropout = torch.nn.Dropout(self.bert_conf.hidden_dropout_prob)

        self.severity_classifier = nn.Linear(in_features=self.bert_conf.hidden_size,
                                                   out_features=config.args.num_severity_labels)
        self.stance_classifier = nn.Linear(in_features=self.bert_conf.hidden_size,
                                                 out_features=config.args.num_stance_labels)
        self.rebuttal_classifier = nn.Linear(in_features=self.bert_conf.hidden_size,
                                                   out_features=config.args.num_rebuttal_labels)

        self.ce_sev_weight = torch.tensor([.1, .2, 1.])
        self.CrossEntropySeverity = CrossEntropyLoss(weight=self.ce_sev_weight)
        self.CrossEntropyStance = CrossEntropyLoss()
        self.CrossEntropyRebuttal = CrossEntropyLoss()

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

        # Metrics
        self.valid_severity_f1_micro = metrics.F1(num_classes=config.args.num_severity_labels, average="micro")
        self.valid_stance_f1_micro = metrics.F1(num_classes=config.args.num_stance_labels, average="micro")
        self.valid_severity_f1_macro = metrics.F1(num_classes=config.args.num_severity_labels, average=None)
        self.valid_stance_f1_macro = metrics.F1(num_classes=config.args.num_stance_labels, average=None)
        self.valid_rebuttal_f1_macro = metrics.F1(num_classes=config.args.num_rebuttal_labels, average=None)



    def forward(self, input_ids=None, masks=None, user_infos=None, token_type_ids=None):
        shared_bert_output = self.shared_bert(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              attention_mask=masks)

        # [batch_size, sequence_length, hidden_size]
        shared_unpooled_output = shared_bert_output[0]

        # [batch_size, hidden_size]
        shared_pooled_output = shared_bert_output[1]
        shared_pooled_output = self.dropout(shared_pooled_output)

        severity_logits = self.severity_classifier(shared_pooled_output)
        stance_logits = self.stance_classifier(shared_pooled_output)
        rebuttal_logits = self.rebuttal_classifier(shared_pooled_output)

        # This will return the indices instead of the values: _ in place of values
        _, severity_predictions = torch.max(severity_logits, 1)
        _, stance_predictions = torch.max(stance_logits, 1)
        _, rebuttal_predictions = torch.max(rebuttal_logits, 1)

        return {"severity_logits":     severity_logits,
               "stance_logits":        stance_logits,
               "rebuttal_logits":      rebuttal_logits,
               "severity_predictions": severity_predictions,
               "stance_predictions":   stance_predictions,
               "rebuttal_predictions": rebuttal_predictions}

    def training_step(self, b_data, batch_idx):
        input_ids = b_data["input_ids"]
        masks = b_data["mask"]
        user_infos = b_data["user_info"]
        severity_labels = b_data["severity_label"]
        stance_labels = b_data["stance_label"]
        rebuttal_labels = b_data["rebuttal_label"]
        topic_labels = b_data["topic_label"]
        theme_labels = b_data["theme_label"]

        # Logits and predictions
        lnps = self.forward(input_ids=input_ids, masks=masks, user_infos=user_infos, token_type_ids=None)

        severity_logits, stance_logits, rebuttal_logits, = lnps["severity_logits"], \
                                                           lnps["stance_logits"], \
                                                           lnps["rebuttal_logits"]
        severity_predictions, stance_predictions, rebuttal_predictions = lnps["severity_predictions"], \
                                                                         lnps["stance_predictions"], \
                                                                         lnps["rebuttal_predictions"]


        sev_loss = self.CrossEntropySeverity(severity_logits, severity_labels)
        st_loss = self.CrossEntropyStance(stance_logits, stance_labels)
        rebut_loss = self.CrossEntropyRebuttal(rebuttal_logits, rebuttal_labels)

        loss = sev_loss + st_loss + rebut_loss
        loss = loss.mean()

        return loss

    def training_epoch_end(self, losses):
        total_loss = 0.0

        for loss in losses:
            total_loss += loss["loss"]
        self.log("tr_loss", total_loss/len(losses), prog_bar=False)

    def validation_step(self, b_data, batch_idx):
        input_ids = b_data["input_ids"]
        masks = b_data["mask"]
        user_infos = b_data["user_info"]
        severity_labels = b_data["severity_label"]
        stance_labels = b_data["stance_label"]
        rebuttal_labels = b_data["rebuttal_label"]
        topic_labels = b_data["topic_label"]
        theme_labels = b_data["theme_label"]

        lnps = self.forward(input_ids=input_ids, masks=masks, user_infos=user_infos, token_type_ids=None)

        severity_logits, stance_logits, rebuttal_logits, = lnps["severity_logits"], \
                                                           lnps["stance_logits"], \
                                                           lnps["rebuttal_logits"]
        severity_predictions, stance_predictions, rebuttal_predictions = lnps["severity_predictions"], \
                                                                         lnps["stance_predictions"], \
                                                                         lnps["rebuttal_predictions"]


        sev_loss = self.CrossEntropySeverity(severity_logits, severity_labels)
        st_loss = self.CrossEntropyStance(stance_logits, stance_labels)
        rebut_loss = self.CrossEntropyRebuttal(rebuttal_logits, rebuttal_labels)

        total_loss = sev_loss + st_loss + rebut_loss

        output = OrderedDict({
            "severity_predictions": severity_predictions,
            "stance_predictions": stance_predictions,
            "rebuttal_predictions": rebuttal_predictions,
            "severity_labels": severity_labels,
            "stance_labels": stance_labels,
            "rebuttal_labels": rebuttal_labels,
            "total_loss": total_loss.mean()
        })

        return output

    def validation_epoch_end(self, outputs):
        total_loss = 0
        severity_predictions = []
        severity_labels = []
        stance_predictions = []
        stance_labels = []
        rebuttal_predictions = []
        rebuttal_labels = []

        for output in outputs:
            severity_predictions.append(output["severity_predictions"])
            severity_labels.append(output["severity_labels"])
            stance_predictions.append(output["stance_predictions"])
            stance_labels.append(output["stance_labels"])
            rebuttal_predictions.append(output["rebuttal_predictions"])
            rebuttal_labels.append(output["rebuttal_labels"])
            total_loss += output["total_loss"]

        severity_predictions = torch.cat(severity_predictions)
        severity_labels = torch.cat(severity_labels)
        stance_predictions = torch.cat(stance_predictions)
        stance_labels = torch.cat(stance_labels)
        rebuttal_predictions = torch.cat(rebuttal_predictions)
        rebuttal_labels = torch.cat(rebuttal_labels)

        real_f1, not_severe_f1, severe_f1 = self.valid_severity_f1_macro(preds=severity_predictions, target=severity_labels)
        self.log("real_f1", real_f1, prog_bar=False)
        self.log("not_severe_f1", not_severe_f1, prog_bar=False)
        self.log("severe_f1", severe_f1, prog_bar=False)

        support_f1, deny_f1, neither_f1 = self.valid_stance_f1_macro(preds=stance_predictions, target=stance_labels)
        self.log("support_f1", support_f1, prog_bar=False)
        self.log("deny_f1", deny_f1, prog_bar=False)
        self.log("neither_f1", neither_f1, prog_bar=False)

        true_f1, false_f1 = self.valid_rebuttal_f1_macro(preds=rebuttal_predictions, target=rebuttal_labels)
        self.log("true_f1", true_f1, prog_bar=False)
        self.log("false_f1", false_f1, prog_bar=False)


        total_loss /= len(outputs)
        self.log("val_loss", total_loss, prog_bar=False)


    def configure_optimizers(self):
        optimizer = transformers.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.config.args.learning_rate, eps=1e-8)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=self.total_steps)
        return [optimizer], [scheduler]
