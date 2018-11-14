from flask import Flask, render_template, request, url_for
from PredictRNNSequenceLabelling import *
import sys, time
import os
from deep.utils import clean_str_teddata
from deep.utils import simple_tokenize

# Initialize the Flask application
app = Flask(__name__)

def load_model(model_dir_prefix):
    """
    """
    model_dir = model_dir_prefix
    start = time.time()
    model = PredictRNNSequenceLabelling(model_dir = model_dir)
    end = time.time()
    print('Loading the model [%s] (%2.2f secs)'%(model_dir,end-start))

    return model
    
# Define a route for the default URL, which loads the form
@app.route('/')
def form():
    return render_template('form_submit.html')

# Define a route for the action of the form, for example '/hello/'
# We are also defining which type of requests this route is 
# accepting: POST requests in this case
@app.route('/punctuation/', methods=['POST'])
def hello():
    sentence=request.form['sentence']
    sentence = sentence.replace('+',' ')
    if not sentence.endswith('</s>'):
        sentence = sentence + ' </s>'

    #start = time.time()
    predictions = app.model.predict([clean_str_teddata(sentence)])
    #duration = time.time() - start

    # score = float(predictions[0][-1])
    # feat = ','.join([str(x) for x in list(predictions[0])])
    # if score > 0.2:
    #     print context, sentence, score
    sentence_punct = ''
    sentence_tokenized = simple_tokenize(sentence)
    if app.model.args.num_classes==4:
        punct_dict = {1:'',2:' ,',3:' .'}
    elif app.model.args.num_classes==3:
        punct_dict = {1:'',2:' .'}

    for word_id,word in enumerate(sentence_tokenized):
         sentence_punct += punct_dict[predictions[0][word_id]] + ' ' + word
    if sentence_punct.endswith('</s>'):
        sentence_punct = sentence_punct[:-5]
    return render_template('form_action.html',sentence_punct=sentence_punct)

# Run the app :)
if __name__ == '__main__':
    app.model_dir_prefix = sys.argv[1]
    app.model = load_model(app.model_dir_prefix)
    #app.run(debug=True)
    print 'flag1'
    app.run(host='0.0.0.0',port=int(sys.argv[2]))
