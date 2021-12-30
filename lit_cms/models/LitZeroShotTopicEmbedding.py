import torch, transformers, random
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from pytorch_lightning import metrics as metrics

from collections import OrderedDict

from pytorch_lightning.core.lightning import LightningModule


class LitZeroShotTopicEmbedding(LightningModule):
    def __init__(self, config, complete_data_dict, training_topics, validation_topics, test_topic):
        super(LitZeroShotTopicEmbedding, self).__init__()
        self.training_topics = training_topics
        self.validation_topics = validation_topics
        self.complete_data_dict = complete_data_dict

        self.total_steps = len(training_topics) * config.args.num_epochs
        self.config = config

        self.bert_config = BertConfig.from_pretrained(config.args.bert_pretrained,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
        self.shared_bert = BertModel.from_pretrained(config.args.bert_pretrained,
                                                     config=self.bert_config)

        self.bert_conf = self.shared_bert.config
        self.dropout = nn.Dropout(self.bert_conf.hidden_dropout_prob)

        # Topic Embedding
        self.reduction_size = 100
        self.tweet_reduction = nn.Linear(in_features=self.bert_conf.hidden_size,
                                         out_features=self.reduction_size)

        # Classifiers
        self.severity_classifier = nn.Linear(in_features=self.bert_conf.hidden_size + self.reduction_size,
                                             out_features=self.config.args.num_severity_labels)
        self.stance_classifier = nn.Linear(in_features=self.bert_conf.hidden_size + self.reduction_size,
                                           out_features=self.config.args.num_stance_labels)
        self.rebuttal_classifier = nn.Linear(in_features=self.bert_conf.hidden_size + self.reduction_size,
                                             out_features=self.config.args.num_rebuttal_labels)

        # Losses
        self.ce_sev_weight = torch.tensor([.1, .2, 1.])
        self.CrossEntropySeverity = CrossEntropyLoss(weight=self.ce_sev_weight)
        self.CrossEntropyStance = CrossEntropyLoss()
        self.CrossEntropyRebuttal = CrossEntropyLoss()
        self.TripletLoss = nn.TripletMarginLoss(margin=1.0)

        # Metrics
        self.valid_severity_f1_micro = metrics.F1(num_classes=config.args.num_severity_labels, average="micro")
        self.valid_stance_f1_micro = metrics.F1(num_classes=config.args.num_stance_labels, average="micro")
        self.valid_severity_f1_macro = metrics.F1(num_classes=config.args.num_severity_labels, average=None)
        self.valid_stance_f1_macro = metrics.F1(num_classes=config.args.num_stance_labels, average=None)
        self.valid_rebuttal_f1_macro = metrics.F1(num_classes=config.args.num_rebuttal_labels, average=None)

    def forward(self, anchor_iid=None, positive_iid=None, negative_iid=None,
                anchor_mask=None, positive_mask=None, negative_mask=None,
                anchor_ttid=None, positive_ttid=None, negative_ttid=None,
                batch_size=4):
        bdim = 0

        if positive_iid is not None:
            input_ids = torch.cat((anchor_iid, positive_iid, negative_iid), bdim)
            attention_mask = torch.cat((anchor_mask, positive_mask, negative_mask), bdim)
            token_type_ids = None

            shared_bert_output = self.shared_bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask)

            shared_pooled_output = shared_bert_output[1]

            anchor_output = shared_pooled_output[:batch_size]

            # -- Check dimensions of this -- #
            reduced_output = self.tweet_reduction(shared_pooled_output)
            #print(f"size: {reduced_output.size()}")
            reduced_tweet = reduced_output[:batch_size]
            reduced_pos = reduced_output[batch_size:batch_size * 2]
            reduced_neg = reduced_output[batch_size * 2:]
            #print(f"Sizes: {reduced_tweet.size(), reduced_pos.size(), reduced_neg.size()}")
            triplet_loss = self.TripletLoss(reduced_tweet, reduced_pos, reduced_neg)

        else:
            shared_bert_output = self.shared_bert(input_ids=anchor_iid,
                                                  token_type_ids=anchor_ttid,
                                                  attention_mask=anchor_mask)

            anchor_output = shared_bert_output[1]
            reduced_tweet = self.tweet_reduction(anchor_output)
            triplet_loss = 0.0
        # [bs, hs] concat [bs, 100] -> [bs, hs+100]
        severity_logits = self.severity_classifier(torch.cat((anchor_output, reduced_tweet), dim=1))
        stance_logits = self.stance_classifier(torch.cat((anchor_output, reduced_tweet), dim=1))
        rebuttal_logits = self.rebuttal_classifier(torch.cat((anchor_output, reduced_tweet), dim=1))

        # This will return the indices instead of the values: _ in place of values
        _, severity_predictions = torch.max(severity_logits, 1)
        _, stance_predictions = torch.max(stance_logits, 1)
        _, rebuttal_predictions = torch.max(rebuttal_logits, 1)

        return {"severity_logits": severity_logits,
                "stance_logits": stance_logits,
                "rebuttal_logits": rebuttal_logits,
                "severity_predictions": severity_predictions,
                "stance_predictions": stance_predictions,
                "rebuttal_predictions": rebuttal_predictions,
                "triplet_loss": triplet_loss}

    def training_step(self, b_data, batch_idx):
        user_infos = b_data["user_info"]
        severity_labels = b_data["severity_label"]
        stance_labels = b_data["stance_label"]
        rebuttal_labels = b_data["rebuttal_label"]
        topic_labels = b_data["topic_label"]
        theme_labels = b_data["theme_label"]

        sampled_topics = self.sample_topics(b_data=b_data,
                                            topic_dict=self.training_topics)

        # Logits and predictions
        lnps = self.forward(anchor_iid=sampled_topics['anchor_iids'],
                            positive_iid=sampled_topics['positive_iids'],
                            negative_iid=sampled_topics['negative_iids'],
                            anchor_mask=sampled_topics['anchor_mask'],
                            positive_mask=sampled_topics['positive_mask'],
                            negative_mask=sampled_topics['negative_mask'],
                            anchor_ttid=None, positive_ttid=None, negative_ttid=None,
                            batch_size=sampled_topics['anchor_iids'].size(0))

        severity_logits, stance_logits, rebuttal_logits, = lnps["severity_logits"], \
                                                           lnps["stance_logits"], \
                                                           lnps["rebuttal_logits"]

        triplet_loss = lnps["triplet_loss"]

        sev_loss = self.CrossEntropySeverity(severity_logits, severity_labels)
        st_loss = self.CrossEntropyStance(stance_logits, stance_labels)
        rebut_loss = self.CrossEntropyRebuttal(rebuttal_logits, rebuttal_labels)

        loss = sev_loss + st_loss + rebut_loss + triplet_loss
        loss = loss.mean()

        return {"loss": loss,
                "triplet_loss": triplet_loss}

    def training_epoch_end(self, outputs):
        total_loss = 0.0
        triplet_losses = 0.0

        for loss in outputs:
            total_loss += loss["loss"]
            triplet_losses += loss["triplet_loss"]
        self.log("tr_loss", total_loss / len(outputs), prog_bar=False)
        self.log("triplet_loss", triplet_losses / len(outputs), prog_bar=False)

    def validation_step(self, b_data, batch_idx):
        input_ids = b_data["input_ids"]
        masks = b_data["mask"]
        user_infos = b_data["user_info"]
        severity_labels = b_data["severity_label"]
        stance_labels = b_data["stance_label"]
        rebuttal_labels = b_data["rebuttal_label"]
        topic_labels = b_data["topic_label"]
        theme_labels = b_data["theme_label"]

        # Logits and predictions
        sampled_topics = self.sample_topics(b_data=b_data,
                                            topic_dict=self.validation_topics)

        # Logits and predictions
        lnps = self.forward(anchor_iid=sampled_topics['anchor_iids'],
                            positive_iid=sampled_topics['positive_iids'],
                            negative_iid=sampled_topics['negative_iids'],
                            anchor_mask=sampled_topics['anchor_mask'],
                            positive_mask=sampled_topics['positive_mask'],
                            negative_mask=sampled_topics['negative_mask'],
                            anchor_ttid=None, positive_ttid=None, negative_ttid=None,
                            batch_size=sampled_topics['anchor_iids'].size(0))

        severity_logits, stance_logits, rebuttal_logits, = lnps["severity_logits"], \
                                                           lnps["stance_logits"], \
                                                           lnps["rebuttal_logits"]
        severity_predictions, stance_predictions, rebuttal_predictions = lnps["severity_predictions"], \
                                                                         lnps["stance_predictions"], \
                                                                         lnps["rebuttal_predictions"]

        triplet_loss = lnps["triplet_loss"]

        sev_loss = self.CrossEntropySeverity(severity_logits, severity_labels)
        st_loss = self.CrossEntropyStance(stance_logits, stance_labels)
        rebut_loss = self.CrossEntropyRebuttal(rebuttal_logits, rebuttal_labels)

        loss = sev_loss + st_loss + rebut_loss + triplet_loss
        total_loss = loss.mean()

        return {"severity_predictions": severity_predictions,
                "stance_predictions": stance_predictions,
                "rebuttal_predictions": rebuttal_predictions,
                "severity_labels": severity_labels,
                "stance_labels": stance_labels,
                "rebuttal_labels": rebuttal_labels,
                "total_loss": total_loss}

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

        real_f1, not_severe_f1, severe_f1 = self.valid_severity_f1_macro(preds=severity_predictions,
                                                                         target=severity_labels)
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
        optimizer = transformers.AdamW([p for p in self.parameters() if p.requires_grad],
                                       lr=self.config.args.learning_rate, eps=1e-8)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=self.total_steps)
        return [optimizer], [scheduler]

    def sample_topics(self, b_data, topic_dict):
        tweet_ids = b_data["tweet_id"]

        b_a_iids = []
        b_a_mask = []
        b_p_iids = []
        b_p_mask = []
        b_n_iids = []
        b_n_mask = []

        topics = set(topic_dict.keys())

        for tid in tweet_ids:
            id = tid.item()
            a_iids = self.complete_data_dict[id]["input_ids"]
            a_mask = self.complete_data_dict[id]["attention_mask"]
            topic_label = self.complete_data_dict[id]["topic"]

            similar_posts = set(topic_dict[topic_label].copy())
            if id in similar_posts:
                similar_posts.remove(id)
            else:
                print('skipped')
            positive_id = random.sample(similar_posts, 1)

            temp_topics = topics.copy()
            temp_topics.remove(topic_label)
            negative_topic = random.sample(temp_topics, 1)
            negative_id = random.sample(topic_dict[negative_topic[0]], 1)

            p_iids = self.complete_data_dict[positive_id[0]]["input_ids"]
            p_mask = self.complete_data_dict[positive_id[0]]["attention_mask"]

            n_iids = self.complete_data_dict[negative_id[0]]["input_ids"]
            n_mask = self.complete_data_dict[negative_id[0]]["attention_mask"]

            b_a_iids.append(a_iids)
            b_a_mask.append(a_mask)
            b_p_iids.append(p_iids)
            b_p_mask.append(p_mask)
            b_n_iids.append(n_iids)
            b_n_mask.append(n_mask)

        # Create them as tensors #
        b_a_iids = torch.cat(b_a_iids, dim=0).to(self.shared_bert.device)
        b_a_mask = torch.cat(b_a_mask, dim=0).to(self.shared_bert.device)
        b_p_iids = torch.cat(b_p_iids, dim=0).to(self.shared_bert.device)
        b_p_mask = torch.cat(b_p_mask, dim=0).to(self.shared_bert.device)
        b_n_iids = torch.cat(b_n_iids, dim=0).to(self.shared_bert.device)
        b_n_mask = torch.cat(b_n_mask, dim=0).to(self.shared_bert.device)

        return {"anchor_iids": b_a_iids,
                "positive_iids": b_p_iids,
                "negative_iids": b_n_iids,
                "anchor_mask": b_a_mask,
                "positive_mask": b_p_mask,
                "negative_mask": b_n_mask}
