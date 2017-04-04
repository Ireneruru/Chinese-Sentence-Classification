#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import gensim
import codecs

def load_data_and_labels(data_path, model_path):

    f = codecs.open(data_path, "rw", encoding="utf8")
    model =  gensim.models.Word2Vec.load(model_path)

    Person = 0
    Location = 0
    Organization = 0
    Number = 0
    Time = 0

    x = []
    Y = []
    y = [0]*5
    for line in f:
        a  = line.split("\t")
        print a
        if a[1] == "Person\n":
            Person = Person + 1
            y[0]=1
        elif a[1] == "Location\n":
            Location = Location + 1
            y[1]=1
        elif a[1] == "Organization\n":
            Organization = Organization + 1
            y[2]=1
        elif a[1] == "Number\n":
            Number = Number + 1
            y[3]=1
        elif a[1] == "Time\n":
            Time = Time + 1
            y[4]=1
        aa = a[0].split(' ')
        sentenceVec = []
        for ll in aa:
            try:
                sentenceVec.append(model[ll])
            except:
                rand = np.random.rand(400)
                sentenceVec.append(rand)
        x.append(sentenceVec)
        Y.append(y)
    print Person, Location, Organization, Number, Time
    y = np.array(Y)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
