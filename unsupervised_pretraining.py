#TODO: try using unsupervised pretraining with encoder decoder
# take descriptions, break them down into words, get the embeddings of the words, encoder decoder to turn it into one vector
# after training that then train the overall model to match those embeddings (more expressive) using distillation -> unsupervised/semisupervised pretraining for any domain
# keep in mind that vectors will change themselves

# train test split
# also think abt masking/other common augs (including maybe keeping old embeddings (unchanged model) and new ones (updating model))

# only train embedding model on most up to date encoded stuff but encoder decoder can be trained on both

#NOTE: data formatting ->  start token, normal tokens, end token, null until end of input size
# [CLS] (start), [SEP] (end), [PAD] (after end)


# 128 tokens
# 128 -> 64 -> 16 -> 4 -> 1 
# 3n expansion not 4n; also use FFN
# keep dont discard but only use encoder