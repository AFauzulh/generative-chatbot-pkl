import os
import json
import pickle

from Model import Model

# Sesuaikan path file lokasi 20k_data
variable_dir = './20k_data'


# ================================================== PENTING, atribut yang dipakai model ==========================================================

with open(variable_dir+'/model_config.json', 'r') as json_file:
    model_config = json.load(json_file)

q_tokenizer_path = variable_dir+'/question_tokenizer.pickle'
a_tokenizer_path = variable_dir+'/answer_tokenizer.pickle'
encoder_weight_path = variable_dir+'/encoder_weights/'
decoder_weight_path = variable_dir+'/decoder_weights/'

BATCH_SIZE = model_config['BATCH_SIZE']
steps_per_epoch = model_config['steps_per_epoch']
embedding_dim = model_config['embedding_dim']
units = model_config['units']
vocab_inp_size = model_config['vocab_inp_size']
vocab_targ_size = model_config['vocab_targ_size']
max_question_len = model_config['max_question_len']
max_answer_len = model_config['max_answer_len']
EPOCHS = model_config['EPOCHS']

==================================================== PENTING =====================================================================================

# Buat Instance Model
model = Model(BATCH_SIZE, steps_per_epoch, embedding_dim, units, vocab_inp_size, vocab_targ_size,
                 max_question_len, max_answer_len, EPOCHS, q_tokenizer_path, a_tokenizer_path, encoder_weight_path,
                 decoder_weight_path)

# Untuk mendapat jawaban tinggal panggil method .respond

print(model.respond("Hello there"))
print(model.respond("Who are you"))
print(model.respond("How are you doing ?"))
print(model.respond("Nice weather, is it ?"))
print(model.respond("I hope you happy"))
print(model.respond("Do you play Black Desert"))
print(model.respond("Hey dude, tell me who is Illezra ?"))