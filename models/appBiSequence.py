from flask import Flask, jsonify, render_template
import time 
from PredictRNNBiSequenceClassification import *
import sys
import os
from deep.utils import clean_str

app = Flask(__name__)

def load_model(model_dir_prefix):
    """
    """
    model_dir = '%s_overfit'%(model_dir_prefix)
    start = time.time()
    model = PredictRNNBiSequenceClassification(model_dir = model_dir)
    end = time.time()
    print('Loading the model [%s] (%2.2f secs)'%(model_dir,end-start))

    return model


@app.route('/api/context_dependent/<context>/<sentence>')
def prediction(context,sentence):
    
    #sentence = sentence.decode('utf8')
    context = context.replace('+',' ')
    sentence = sentence.replace('+',' ')

    #start = time.time()
    predictions = app.model.predict([clean_str(context)],[clean_str(sentence)])
    #duration = time.time() - start

    score = float(predictions[0][-1])
    feat = ','.join([str(x) for x in list(predictions[0])])
    if score > 0.2:
        print context, sentence, score
    """
    print sentence
    print predictions
    response = {'score' :  float(predictions[0][1]), 'text' : sentence, 'time_taken_secs' : duration}
    return jsonify(response)
    """
    return feat
 
if __name__ == "__main__":
    

    app.model_dir_prefix = sys.argv[1]
    app.model = load_model(app.model_dir_prefix)
    #app.run(debug=True)
    app.run(host='0.0.0.0',port=int(sys.argv[2]))
