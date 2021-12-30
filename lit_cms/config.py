import argparse
from lit_cms.models.LitBERT import LitBERT
from lit_cms.models.LitZeroShotTopicEmbedding import LitZeroShotTopicEmbedding

class Config:
    def __init__(self):
        '''
        Possible Experiments
            LitZeroShotTopicEmbedding
            BERT
            TwitterBERT
        '''
        experiment = "LitBERT"

        parser = argparse.ArgumentParser(description="data and model types")

        # Configurations for the models
        parser.add_argument('--num_user_info', default=7,
                            required=False, type=int,
                            help='number of user information arguments')
        parser.add_argument('--num_severity_labels', default=3,
                            required=False, type=int,
                            help='number of labels used for severity classification')
        parser.add_argument('--num_stance_labels', default=3,
                            required=False, type=int,
                            help='number of labels used for stance classification')
        parser.add_argument('--num_rebuttal_labels', default=2,
                            required=False, type=int,
                            help='number of labels used for rebuttal classification')
        parser.add_argument('--batch_size', default=4,
                            required=False, type=int,
                            help='the batch size')
        parser.add_argument('--learning_rate', default=2e-5,
                            required=False, type=float,
                            help='the learning rate')
        parser.add_argument('--num_epochs', default=20,
                            required=False, type=int,
                            help='the number of epochs to run the model for')
        parser.add_argument('--seed', default=79,
                            required=False, type=int,
                            help='starting seed for all randoms')

        # Data Files
        parser.add_argument('--merged_tweet_file', default='data/unmerged/all_merged_data.json',
                            required=False, type=str,
                            help='the entire tweet set merged for convenience')
        parser.add_argument('--tweet_annotation_file', default='data/last_pull.json',
                            required=False, type=str,
                            help='the file with the ids and annotations')

        # Model Inputs

        parser.add_argument('--zs_input_model',
                            default='/shared/hltdir4/disk1/team/data/models/bert/oa-cov/zero-shot/zs-79/ckpt/ckpt-3.pt',
                            required=False, type=str,
                            help='the trained for the zero shot part of this model')

        # Model Outputs
        parser.add_argument('--output_directory', default='/shared/hltdir4/disk1/team/data/models/bert/oa-cov/sestre/',
                            required=False, type=str,
                            help='the base output directory to store the model')
        parser.add_argument('--model2run', default=experiment,
                            required=False, type=str,
                            help='the model we are running as well as the correct output directory')
        parser.add_argument('--output_tokenizer_directory',
                            default='/shared/hltdir4/disk1/team/data/models/bert/oa-cov/tokenizer/',
                            required=False, type=str,
                            help='the output directory for the tokenizer')

        # Which Experiment to run
        if experiment == "LitZeroShotTopicEmbedding":
            parser.add_argument('--experiment_name', default='Lit/ZeroShotTopicEmbedding',
                                required=False, type=str,
                                help='the zero shot model for topic embedding')
            parser.add_argument('--project_name', default='omeedashtiani/covid-lightning',
                                required=False, type=str,
                                help='the experiment name to save the data for zero shot topic embeddings')
            parser.add_argument('--bert_pretrained', default='bert-base-uncased',
                                required=False, type=str,
                                help='the bert pretrained model to used')
        elif experiment == "LitBERT":
            parser.add_argument('--experiment_name', default='Lit/BERT',
                                required=False, type=str,
                                help='the zero shot model for topic embedding')
            parser.add_argument('--project_name', default='omeedashtiani/covid-lightning',
                                required=False, type=str,
                                help='plain BERT')
            parser.add_argument('--bert_pretrained', default='bert-base-uncased',
                                required=False, type=str,
                                help='the bert pretrained model to used')
        elif experiment == "TwitterBERT":
            parser.add_argument('--experiment_name', default='Lit/TwitterBERT',
                                required=False, type=str,
                                help='Using a pretrained covid twitter BERT')
            parser.add_argument('--project_name', default='omeedashtiani/covid-lightning',
                                required=False, type=str,
                                help='Save twitter BERT details')
            parser.add_argument('--bert_pretrained',
                                default='/shared/hltdir4/disk1/team/data/models/bert/covid-twitter-bert/',
                                required=False, type=str,
                                help='directory of the twitter pretrained model')



        self.args = parser.parse_args()

def GetModel(config, complete_data_dict, training_topics, validation_topics, test_topics):
    model2run = config.args.model2run
    print(f"Running model: {model2run}")

    if model2run == "LitBERT" or model2run == 'TwitterBERT':
        return LitBERT(config, complete_data_dict, training_topics, validation_topics, test_topics)
    elif model2run == "LitZeroShotTopicEmbedding":
        return LitZeroShotTopicEmbedding(config, complete_data_dict, training_topics, validation_topics, test_topics)
