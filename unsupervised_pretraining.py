# training using triplet loss and Matryoshka embeddings
# starting model is "all-MiniLM-L6-v2"

#TODO: imports


#TODO: collect book_ids that are non null in goodreads books + at least 14 reviews



#TODO: create dataloader (grabs up to 64 positives and 64+ negatives)


#TODO: train with ohem (worst negatives and positives, batch size 128 -> useful samples (harder than a threshold)


    # after each round, lets validate with a val set 
        # make sure that same groups have higher similarity than random


#TODO: final testing set is the same
