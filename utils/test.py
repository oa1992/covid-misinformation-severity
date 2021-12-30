import data_utils as du

#tweet_dict = du.create_tweet_dict('../data/unmerged/annod_data.json')
#print(du.create_twitter_graph(tweet_dict))
#print(tweet_dict)
tweet_labels = du.create_tweet_label_dict('../data/labeled_misinformation_real_dupeless.json')
#print(tweet_labels)
#du.count_stats_labels(tweet_labels)
#
# tweet ids with all of their information
tweet_file = '../data/unmerged/annod_data.json'
tweet_dict = du.create_tweet_dict(tweet_file)
#du.print_rebuttal_tweets(tweet_dict, tweet_labels)

du.return_x_each(tweet_dict, tweet_labels, 20)

#sentences, user_infos, sev_labs, st_labs, rebut_labs = du.retrieve_anno_ids(tweet_labels, tweet_dict)
