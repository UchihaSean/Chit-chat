# -*- coding: UTF-8 -*-
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
import csv
import numpy as np
class seq2seq():
    def __init__(self, sequence_length = 50, batch_size = 128, hidden_size = 256, num_layers=2, num_encoder_symbbols = 5004,
                 num_decoder_symbols=5004, embedding_size = 256, learning_rate = 0.001):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_encoder_symbols = num_encoder_symbbols  # 'UNK' and '<go>' and '<eos>' and '<pad>'
        self.num_decoder_symbols = num_decoder_symbols
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate


    def ask_question(self, question):
        if len(question)>self.sequence_length:
            question=question[:self.sequence_length]
        graph = tf.Graph()
        with graph.as_default():
            model = Seq2SeqModel(self.hidden_size, self.num_layers, self.batch_size, self.sequence_length, self.embedding_size,
                                 self.learning_rate, self.num_encoder_symbols, self.num_decoder_symbols, 'true')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as session:
                reply = model.test(session, question, epoch=26)

        return reply
def seq2seq_output(input_file_name, output_file_name):
    """
    file output
    """
    input_file = open(input_file_name)
    questions = []
    for i,line in enumerate(input_file.readlines()):
        if i % 2==0:
            questions.append(line.split(" ")[-1].strip().decode("utf-8"))
        if i>10000: break
    seq =seq2seq()

    with open(output_file_name,'w') as csvfile:
        filednames = ['Question','Reply','Score']
        writer = csv.DictWriter(csvfile, fieldnames=filednames)
        writer.writeheader()
        for i in range(100):
            print(i)
            t = int(np.random.random()*len(questions))
            print(questions[t])
            dict = {'Score':"", 'Question':questions[t].encode("utf-8")}
            reply = seq.ask_question(questions[t])
            print(reply)
            dict["Reply"] = reply.encode("utf-8")
            writer.writerow(dict)






def main():
    seq2seq_output("data/jd_chat.txt","data/seq2seq_output.csv")
    # seq = seq2seq()
    # print(seq.ask_question("今天多久能到"))

if __name__ == "__main__":
    main()