import os
import pdb
import random
from text_data import loadPrepareData
from text_data import indexesFromSentence
from text_data import batch2TrainData
import torch.nn as nn
import torch

DATA_DIR = "/home/changmin/research/steganography/data/"
TEXT = "dialogues_text.txt"
ALL_PATH = os.path.join(DATA_DIR, "dialogues_text.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "train/dialogues_train.txt")
VAL_PATH = os.path.join(DATA_DIR, "validation/dialogues_validation.txt")
TEST_PATH = os.path.join(DATA_DIR, "test/dialogues_test.txt")


voc, pairs = loadPrepareData(None, "dialog", TRAIN_PATH, 768)
#print(voc.word2index)
#print(pairs[0][0])
#print(indexesFromSentence(voc, pairs[0][0]))

batch_size = 32
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

embedding = nn.Embedding(voc.num_words, 256)
print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
print(input_variable.shape)
test = input_variable.view(batch_size, 3, 256)
print(embedding(input_variable).shape)
print(embedding(test).shape)
print(embedding(input_variable).view(batch_size, 3, 256, 256).shape)

