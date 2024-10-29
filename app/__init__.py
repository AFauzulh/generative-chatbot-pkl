import os
import sys
from flask import Flask, render_template, request , session
sys.path.append(os.path.abspath("/chatbot-pkl-docker"))
from Model import Model

import json
import pickle

# Sesuaikan path file lokasi 20k_data
# variable_dir = '/code/program/20k_data'
variable_dir = '/chatbot-pkl-docker/variable_dump'


with open(variable_dir+'/model_config.json', 'r') as json_file:
    model_config = json.load(json_file)

q_tokenizer_path = variable_dir+'/question_tokenizer.pickle'
a_tokenizer_path = variable_dir+'/answer_tokenizer.pickle'
encoder_weight_path = variable_dir+'/encoder_weights/'
decoder_weight_path = variable_dir+'/decoder_weights/'

BATCH_SIZE = model_config['BATCH_SIZE']
steps_per_epoch = model_config['steps_per_epoch']
val_steps_per_epoch = model_config['val_steps_per_epoch']
embedding_dim = model_config['embedding_dim']
units = model_config['units']
vocab_inp_size = model_config['vocab_inp_size']
vocab_targ_size = model_config['vocab_targ_size']
max_question_len = model_config['max_question_len']
max_answer_len = model_config['max_answer_len']
EPOCHS = model_config['EPOCHS']

# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.secret_key = 'bangkit'

# Route to fast check the app
@app.route('/hello')
def hello():
    return 'Hello, World!'

    # Open the chat app
@app.route("/")
def index():
    session['lastText'] = ' '
    session['repetition'] = '0'
    return render_template("index.html")

    # Get the response from chatbot
@app.route("/get", methods=["GET"])
def get_response():
    userText = str(request.args.get('msg'))
    if session['lastText'] == userText:
        repetition = int(session['repetition']) + 1
        question_term = ['what', 'why', 'how', 'when', 'who', '?', 'where']
        userText = userText.lower()
        question_sentence = any(question_term in userText for question_term in question_term)
        if repetition == 1 and question_sentence:
            hasil = 'Why you ask the same question?'
        elif repetition == 1:
            hasil = 'I think you already said it'
        elif repetition == 2:
            hasil = 'Is that all you can say? even i have more words than you and i am just a computer'
        elif repetition == 3:
            hasil = 'Are you broken?'
        elif repetition >= 4:
            hasil = 'I think you still broken, please tell me something else'
        session['repetition'] = str(repetition)
        return hasil
    else:    
        session['lastText'] = userText
        session['repetition'] = '0'
            
        model = Model(BATCH_SIZE, steps_per_epoch, val_steps_per_epoch, embedding_dim, units, vocab_inp_size, vocab_targ_size,
        max_question_len, max_answer_len, EPOCHS, q_tokenizer_path, a_tokenizer_path, encoder_weight_path,
        decoder_weight_path)
        return str(model.respond(userText))   
    
@app.route("/cache", methods=["GET"])
def get_cache():
    return session['repetition']

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8998, debug=True)

# def create_app(test_config=None):
#     # create and configure the app
#     app = Flask(__name__, instance_relative_config=True)
#     app.secret_key = 'bangkit'

#     # Route to fast check the app
#     @app.route('/hello')
#     def hello():
#         return 'Hello, World!'

#     # Open the chat app
#     @app.route("/")
#     def index():
#         session['lastText'] = ' '
#         session['repetition'] = '0'
#         return render_template("index.html")

#     # Get the response from chatbot
#     @app.route("/get", methods=["GET"])
#     def get_response():
#         userText = str(request.args.get('msg'))
#         if session['lastText'] == userText:
#             repetition = int(session['repetition']) + 1
#             question_term = ['what', 'why', 'how', 'when', 'who', '?', 'where']
#             userText = userText.lower()
#             question_sentence = any(question_term in userText for question_term in question_term)
#             if repetition == 1 and question_sentence:
#                 hasil = 'Why you ask the same question?'
#             elif repetition == 1:
#                 hasil = 'I think you already said it'
#             elif repetition == 2:
#                 hasil = 'Is that all you can say? even i have more words than you and i am just a computer'
#             elif repetition == 3:
#                 hasil = 'Are you broken?'
#             elif repetition >= 4:
#                 hasil = 'I think you still broken, please tell me something else'
#             session['repetition'] = str(repetition)
#             return hasil
#         else:    
#             session['lastText'] = userText
#             session['repetition'] = '0'
            
#             model = Model(BATCH_SIZE, steps_per_epoch, val_steps_per_epoch, embedding_dim, units, vocab_inp_size, vocab_targ_size,
#             max_question_len, max_answer_len, EPOCHS, q_tokenizer_path, a_tokenizer_path, encoder_weight_path,
#             decoder_weight_path)
#             return str(model.respond(userText))   
    
#     @app.route("/cache", methods=["GET"])
#     def get_cache():
#         return session['repetition']

#     return app

