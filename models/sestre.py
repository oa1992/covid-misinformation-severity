# Severity, Stance, and Rebuttal Model when being passed the
# Topic/Theme embeddings.
import torch
from transformers import BertModel, BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from models.Attention import AttentionPooling
from models.SelectedSharingLayer import SelectedSharingLayer
from models.MultiHeadAttention import MultiHeadAttention


class StanceRebuttal(BertPreTrainedModel):
    def __init__(self, config, num_user_info=7, num_severity=3, num_stances=3, num_rebuttals=2, num_themes=6):
        super(StanceRebuttal, self).__init__(config)



        # Bert must be passed its config
        #print(config)
        #self.BPTM = BertPreTrainedModel.from_pretrained(config, output_attentions=False, output_hidden_states=False)
        self.bert = BertModel(config)
        #self.BPTM.init_weights()
        #self.bert = BertModel(self.BPTM.config)#.from_pretrained('bert-base-uncased',
                              #                output_attentions=False,
                              #                output_hidden_states=False)

        #self.bert.init_weights()

        # Dropout
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Data information
        self.num_user_info = num_user_info
        self.num_severity = num_severity
        self.num_stance = num_stances
        self.num_rebuttals = num_rebuttals

        # Classifiers for each thingy
        #--------Test 0--------#
        self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_severity)
        self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_stances)
        self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)
        # self.severity_classifier = torch.nn.Linear(in_features=768, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=768, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=768, out_features=num_rebuttals)
        #--------Test 1--------#
        # self.severity_classifier = torch.nn.Linear(in_features=100, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=100, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=100, out_features=num_rebuttals)

        #--------Test 2--------#
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_rebuttals)
        # self.theme_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_themes)

        #--------Test 3---------#
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=config.hidden_size)
        # self.attention_pooling_st = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=config.hidden_size)
        # self.attention_pooling_re = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=config.hidden_size)
        #
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)

        #--------Test 4---------#
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=config.hidden_size)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)

        #--------Test 5---------#
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_rebuttals)

        #self.theme_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=6)

        #--------Test 6---------#
        # self.sifted = SelectedSharingLayer(merged_size=config.hidden_size*2, seq_len=200)
        # self.bert_stance = BertModel(config)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size*9*200, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size*9*200, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)

        #self.theme_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_themes)

        # --------Test 7---------#
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size * 2 + 100, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size + 100, out_features=num_rebuttals)

        # --------Test 8---------#
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.attention_pooling_st = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.sifted = SelectedSharingLayer(merged_size=config.hidden_size*2, seq_len=200)
        # self.bert_stance = BertModel(config)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size*9*200+config.hidden_size, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size*9*200+config.hidden_size, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)

        # --------Test 9---------# Run MH attn with topic embeddings
        # self.bert_stance = BertModel(config)
        # self.sifted = SelectedSharingLayer(merged_size=config.hidden_size * 3, seq_len=200)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size*14*200, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size*14*200, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)

        # --------Test 10--------#
        # self.bert_stance = BertModel(config)
        # self.bert_combined = BertModel(config)
        # self.sifted = SelectedSharingLayer(merged_size=config.hidden_size, seq_len=200)
        # self.attention_pooling_sev = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.attention_pooling_st = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.attention_pooling_comb = AttentionPooling(hidden_size=config.hidden_size, topic_embedding_size=100)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size * 5 * 200 + config.hidden_size*2,
        #                                            out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size * 5 * 200 + config.hidden_size*2,
        #                                            out_features=num_severity)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size,
        #                                            out_features=num_severity)

        # ---------Test 11-----------#
        # self.sifted = SelectedSharingLayer(merged_size=config.hidden_size, seq_len=200)
        # self.bert_stance = BertModel(config)
        # self.severity_classifier = torch.nn.Linear(in_features=config.hidden_size*5*200, out_features=num_severity)
        # self.stance_classifier = torch.nn.Linear(in_features=config.hidden_size*5*200, out_features=num_stances)
        # self.rebuttal_classifier = torch.nn.Linear(in_features=config.hidden_size, out_features=num_rebuttals)


        #self.ce_sev_weight = torch.tensor([1., 1., 1.]).to("cuda")#[.1, .2, 1.]).to("cuda")
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, topic_embedding=None, user_infos=None,
                      sev_labels=None, st_labels=None, rebut_labels=None, topic_labels=None, theme_labels=None,
                      batch_size=4):
        """
        Given topic embeddings from a previous model (and user information)
            Find stance
            Find severity
            Find rebuttals

        :param topic_embedding:
        :param user_infos:
        :param sev_labels:
        :param st_labels:
        :param rebut_labels:
        :param topic_labels:
        :param batch_size:
        :return:
        """

        ############################################################################
        ###       Reduce the size of the embedding to a smaller dimension        ###
        ############################################################################
        # reduced_tweet =  self.tweet_reduce(pooled_tweet)
        # reduced_pos   =  self.pos_reduce(pooled_pos)
        # reduced_neg   =  self.neg_reduce(pooled_neg)
        bert_output = self.bert(input_ids, token_type_ids, attention_mask)
        # [batch_size, sequence_length, hidden_size]
        unpooled_output = bert_output[0]
        # [batch_size, hidden_size]
        pooled_output = bert_output[1]

        pooled_output = self.dropout(pooled_output)


        #top_user = torch.cat((topic_embedding, user_infos), dim=1)
        #out_user = torch.cat((pooled_output, user_infos), dim=1)
        #top_user = topic_embedding
        #------- TEST 0 -------# bert_only.txt
        sev_logits = self.severity_classifier(pooled_output)
        st_logits = self.stance_classifier(pooled_output)
        rebut_logits = self.rebuttal_classifier(pooled_output)


        #------- TEST 1 -------# topic_embedding
        # sev_logits = self.severity_classifier(topic_embedding)
        # st_logits = self.stance_classifier(topic_embedding)
        # rebut_logits = self.rebuttal_classifier(topic_embedding)

        #------- TEST 2 -------# top_emb_cat_pooled_output
        # pte = torch.cat((pooled_output, topic_embedding), dim=1)
        # sev_logits = self.severity_classifier(pte)
        # st_logits = self.stance_classifier(pte)
        # rebut_logits = self.rebuttal_classifier(pte)
        #theme_logits = self.theme_classifier(pte)

        #------- Test 3 -------# top emb cat with pooled outputs and then attention
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                        queries=topic_embedding,
        #                                        attention_mask=attention_mask)
        # attended_info_st = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                            queries=topic_embedding,
        #                                            attention_mask=attention_mask)
        # attended_info_re = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                            queries=topic_embedding,
        #                                            attention_mask=attention_mask)
        # sev_logits = self.severity_classifier(attended_info_sev)
        # st_logits = self.stance_classifier(attended_info_st)
        # rebut_logits = self.rebuttal_classifier(attended_info_re)

        #-------- Test 4 ------# Only use attention for severity, use bert for st and re
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                                queries=topic_embedding,
        #                                                attention_mask=attention_mask)
        # sev_logits = self.severity_classifier(attended_info_sev)
        # st_logits = self.stance_classifier(pooled_output)
        # rebut_logits = self.rebuttal_classifier(pooled_output)

        #-------- Test 5 ------# Attention for severity, concatenation for ST and RE
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                                queries=topic_embedding,
        #                                                attention_mask=attention_mask)
        # pte = torch.cat((pooled_output, topic_embedding), dim=1)
        # sev_logits = self.severity_classifier(attended_info_sev)
        # st_logits = self.stance_classifier(pte)
        # rebut_logits = self.rebuttal_classifier(pte)

        #theme_logits = self.theme_classifier(pte)

        #-------- Test 6 -------# BEST SO FAR
        # bert_output_stance = self.bert_stance(input_ids, token_type_ids, attention_mask)
        # unpooled_stance, pooled_stance = bert_output_stance[0], bert_output_stance[1]
        # concated_berts = torch.cat((unpooled_output, unpooled_stance), -1)
        # SSL_severity = self.sifted(Efake=unpooled_output, Hshared=concated_berts)
        # SSL_stance = self.sifted(Efake=unpooled_stance, Hshared=concated_berts)
        #
        # cat_sev_SSL = torch.cat((unpooled_output, SSL_severity), dim=-1)
        # szs = cat_sev_SSL.size()
        # cat_sev_SSL = cat_sev_SSL.view(szs[0], szs[1] * szs[2])
        #
        # cat_st_SSL = torch.cat((unpooled_stance, SSL_stance), dim=-1)
        # cat_st_SSL = cat_st_SSL.view(szs[0], szs[1] * szs[2])
        #
        # sev_logits = self.severity_classifier(cat_sev_SSL)
        # st_logits = self.stance_classifier(cat_st_SSL)
        # rebut_logits = self.rebuttal_classifier(pooled_output)

        # theme_logits = self.theme_classifier(pooled_output)

        #-------- Test 7 -------# GARBAGE IT NO WORK
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                                queries=topic_embedding,
        #                                                attention_mask=attention_mask)
        # pte = torch.cat((pooled_output, topic_embedding), dim=1)
        # pte_attn = torch.cat((attended_info_sev, pte), dim=1)
        # sev_logits = self.severity_classifier(pte_attn)
        # st_logits = self.stance_classifier(pte)
        # rebut_logits = self.rebuttal_classifier(pte)

        #-------- Test 8 -------#
        # bert_output_stance = self.bert_stance(input_ids, token_type_ids, attention_mask)
        # unpooled_stance, pooled_stance = bert_output_stance[0], bert_output_stance[1]
        # concated_berts = torch.cat((unpooled_output, unpooled_stance), -1)
        # SSL_severity = self.sifted(Efake=unpooled_output, Hshared=concated_berts)
        # SSL_stance = self.sifted(Efake=unpooled_stance, Hshared=concated_berts)
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                                queries=topic_embedding,
        #                                                attention_mask=attention_mask)
        # attended_info_st = self.attention_pooling_st(hidden_states=unpooled_stance,
        #                                              queries=topic_embedding,
        #                                              attention_mask=attention_mask)
        #
        # cat_sev_SSL = torch.cat((unpooled_output, SSL_severity), dim=-1)
        # szs = cat_sev_SSL.size()
        # cat_sev_SSL = cat_sev_SSL.view(szs[0], szs[1] * szs[2])
        # cat_sev_SSL = torch.cat((cat_sev_SSL, attended_info_sev), dim=-1)
        #
        # cat_st_SSL = torch.cat((unpooled_stance, SSL_stance), dim=-1)
        # cat_st_SSL = cat_st_SSL.view(szs[0], szs[1] * szs[2])
        # cat_st_SSL = torch.cat((cat_st_SSL, attended_info_st), dim=-1)
        #
        # sev_logits = self.severity_classifier(cat_sev_SSL)
        # st_logits = self.stance_classifier(cat_st_SSL)
        # rebut_logits = self.rebuttal_classifier(pooled_output)

        # -------- Test 9 -------# no mem
        # bert_output_stance = self.bert_stance(input_ids, token_type_ids, attention_mask)
        # unpooled_stance, pooled_stance = bert_output_stance[0], bert_output_stance[1]
        # unpooled_topemb_sev = torch.cat((topic_embedding, unpooled_output), dim=-1)
        # unpooled_topemb_st = torch.cat((topic_embedding, unpooled_stance), dim=-1)
        #
        # concated_berts = torch.cat((unpooled_topemb_sev, unpooled_stance), dim=-1)
        #
        # SSLTopEmbSev = self.sifted(Efake=unpooled_topemb_sev, Hshared=concated_berts)
        # SSLTopEmbSt = self.sifted(Efake=unpooled_topemb_st, Hshared=concated_berts)
        #
        # cat_sev_SSL = torch.cat((unpooled_topemb_sev, SSLTopEmbSev), dim=-1)
        # szs = cat_sev_SSL.size()
        # cat_sev_SSL = cat_sev_SSL.view(szs[0], szs[1] * szs[2])
        #
        # cat_st_SSL = torch.cat((unpooled_topemb_st, SSLTopEmbSt), dim=-1)
        # cat_st_SSL = cat_st_SSL.view(szs[0], szs[1] * szs[2])
        #
        # sev_logits = self.severity_classifier(cat_sev_SSL)
        # st_logits = self.stance_classifier(cat_st_SSL)
        # rebut_logits = self.rebuttal_classifier(pooled_output)

        # --------- Test 10 --------# Yolo
        # bdim=-1
        # comb_iids, comb_ttids, comb_ams = torch.cat((input_ids, input_ids), dim=bdim), \
        #                                     None, \
        #                                   torch.cat((attention_mask, attention_mask), dim=bdim)
        #
        # bert_output_stance = self.bert_stance(input_ids, token_type_ids, attention_mask)
        # unpooled_stance, pooled_stance = bert_output_stance[0], bert_output_stance[1]
        # bert_comb = self.bert_combined(comb_iids, comb_ttids, comb_ams)
        # up_bert_comb, po_bert_comb = bert_comb[0], bert_comb[1]
        # print(up_bert_comb.size())
        #
        # SSL_severity = self.sifted(Efake=unpooled_output, Hshared=up_bert_comb)
        # SSL_stance = self.sifted(Efake=unpooled_stance, Hshared=up_bert_comb)
        #
        # attended_info_sev = self.attention_pooling_sev(hidden_states=unpooled_output,
        #                                                queries=topic_embedding,
        #                                                attention_mask=attention_mask)
        # attended_info_st = self.attention_pooling_st(hidden_states=unpooled_stance,
        #                                              queries=topic_embedding,
        #                                              attention_mask=attention_mask)
        # attended_info_comb = self.attention_pooling_comb(hidden_states=up_bert_comb,
        #                                              queries=topic_embedding,
        #                                              attention_mask=attention_mask)
        #
        # cat_sev_SSL = torch.cat((unpooled_output, SSL_severity), dim=-1)
        # szs = cat_sev_SSL.size()
        # cat_sev_SSL = cat_sev_SSL.view(szs[0], szs[1] * szs[2])
        # cat_sev_SSL = torch.cat((cat_sev_SSL, attended_info_sev), dim=-1)
        # cat_sev_SSL = torch.cat((cat_sev_SSL, attended_info_comb), dim=-1)
        #
        # cat_st_SSL = torch.cat((unpooled_stance, SSL_stance), dim=-1)
        # cat_st_SSL = cat_st_SSL.view(szs[0], szs[1] * szs[2])
        # cat_st_SSL = torch.cat((cat_st_SSL, attended_info_st), dim=-1)
        # cat_st_SSL = torch.cat((cat_st_SSL, attended_info_comb), dim=-1)
        #
        # sev_logits = self.severity_classifier(cat_sev_SSL)
        # st_logits = self.stance_classifier(cat_st_SSL)
        # rebut_logits = self.rebuttal_classifier(pooled_output)

        #--------------Test 11---------------#
        # bert_output_stance = self.bert_stance(input_ids, token_type_ids, attention_mask)
        # unpooled_stance, pooled_stance = bert_output_stance[0], bert_output_stance[1]
        # #concated_berts = torch.cat((unpooled_output, unpooled_stance), -1)
        # SSL_severity = self.sifted(Efake=unpooled_output, Hshared=unpooled_stance)
        # SSL_stance = self.sifted(Efake=unpooled_stance, Hshared=unpooled_output)
        #
        # cat_sev_SSL = torch.cat((unpooled_output, SSL_severity), dim=-1)
        # szs = cat_sev_SSL.size()
        # cat_sev_SSL = cat_sev_SSL.view(szs[0], szs[1] * szs[2])
        #
        # cat_st_SSL = torch.cat((unpooled_stance, SSL_stance), dim=-1)
        # cat_st_SSL = cat_st_SSL.view(szs[0], szs[1] * szs[2])
        #
        # sev_logits = self.severity_classifier(cat_sev_SSL)
        # st_logits = self.stance_classifier(cat_st_SSL)
        # rebut_logits = self.rebuttal_classifier(pooled_output)





        ce_sev_loss_fct = CrossEntropyLoss()#weight=self.ce_sev_weight)
        ce_loss_fct = CrossEntropyLoss()

        if sev_labels is not None:
            sev_loss = ce_sev_loss_fct(sev_logits, sev_labels)
            st_loss = ce_loss_fct(st_logits, st_labels)
            rebut_loss = ce_loss_fct(rebut_logits, rebut_labels)
            #theme_loss = ce_loss_fct(theme_logits, theme_labels)

            #total_loss = sev_loss + st_loss + rebut_loss + triplet_loss
            total_loss = sev_loss + st_loss + rebut_loss# + theme_loss
            # This will return the indices instead of the values: _ in place of values
            _, sev_preds = torch.max(sev_logits, 1)
            _, st_preds = torch.max(st_logits, 1)
            _, rebut_preds = torch.max(rebut_logits, 1)
            #_, theme_preds = torch.max(theme_logits, 1)

            total_loss = total_loss.mean()
            return sev_preds, st_preds, rebut_preds,  total_loss
            # return sev_preds, st_preds, rebut_preds, sev_loss, st_loss, rebut_loss
        else:
            # This will return the indices instead of the values: _ in place of values
            _, sev_preds = torch.max(sev_logits, 1)
            _, st_preds = torch.max(st_logits, 1)
            _, rebut_preds = torch.max(rebut_logits, 1)
            #_, theme_preds = torch.max(theme_logits, 1)

            return sev_preds, st_preds, rebut_preds,


