import pickle
import re

import numpy as np
import tensorflow as tf

from Encoder import Encoder
from BahdanauAttention import BahdanauAttention
from Decoder import Decoder

class Model():
    def __init__(self, BATCH_SIZE, steps_per_epoch, val_steps_per_epoch, embedding_dim, units, vocab_inp_size, vocab_targ_size,
                 max_question_len, max_answer_len, EPOCHS, q_tokenizer_path, a_tokenizer_path, encoder_weight_path,
                 decoder_weight_path):
        self.BATCH_SIZE = BATCH_SIZE
        self.steps_per_epoch = steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_inp_size = vocab_inp_size
        self.vocab_targ_size = vocab_targ_size
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.EPOCHS = EPOCHS
        self.q_tokenizer_path = q_tokenizer_path
        self.a_tokenizer_path = a_tokenizer_path
        self.encoder_weight_path = encoder_weight_path

        self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_targ_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.attention = BahdanauAttention(10)

        self.encoder.load_weights(filepath=encoder_weight_path)
        self.decoder.load_weights(filepath=decoder_weight_path)

        with open(q_tokenizer_path, 'rb') as f:
            self.inp_lang_tokenizer = pickle.load(f)
        with open(a_tokenizer_path, 'rb') as f:
            self.targ_lang_tokenizer = pickle.load(f)

    def evaluate(self, sentence):
        attention_plot = np.zeros((self.max_answer_len, self.max_question_len))

        sentence = self._clean_text(sentence)
        sentence = self._remove_number(sentence)
        sentence = self._preprocess_sentence(sentence)

        inputs = []

        for i in sentence.split(' '):
            if i not in self.inp_lang_tokenizer.index_word.values():
                inputs.append(self.inp_lang_tokenizer.word_index['<OOV>'])
            else:
                inputs.append(self.inp_lang_tokenizer.word_index[i])

        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.max_question_len, padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang_tokenizer.word_index['<start>']], 0)

        for t in range(self.max_answer_len):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.targ_lang_tokenizer.index_word[predicted_id] + ' '

            if self.targ_lang_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def respond(self, sentence):
        sentence = sentence.lower()
        result, _, _ = self.evaluate(sentence)
        return result

    def _preprocess_sentence(self, data):
        return '<start> ' + data + ' <end>'

    def _remove_number(self, data):
        return re.sub(r'\d+', '', data)

    def _clean_text(self, txt):
        txt = txt.lower()
        txt = re.sub(r"i'm", "i am", txt)
        txt = re.sub(r"he's", "he is", txt)
        txt = re.sub(r"she's", "she is", txt)
        txt = re.sub(r"that's", "that is", txt)
        txt = re.sub(r"what's", "what is", txt)
        txt = re.sub(r"where's", "where is", txt)
        txt = re.sub(r"\'ll", " will", txt)
        txt = re.sub(r"\'ve", " have", txt)
        txt = re.sub(r"\'re", " are", txt)
        txt = re.sub(r"\'d", " would", txt)
        txt = re.sub(r"won't", "will not", txt)
        txt = re.sub(r"can't", "can not", txt)
        txt = re.sub(r"u", "you", txt)
        txt = re.sub(r"[^\w\s]", "", txt)
        return txt
