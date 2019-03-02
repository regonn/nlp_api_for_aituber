from flask import Flask, request, jsonify
from sentiment_analysis import SentimentAnalysis

SA_TOKENIZER = './models/tokenizer.pkl'
SA_MODEL = './models/model.h5'

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global sentiment_analysis

    sentiment_analysis = SentimentAnalysis(SA_TOKENIZER, SA_MODEL)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.route('/analyze_sentiment')
def analyze_sentiment():
    if not sentiment_analysis:
        _load_model()
        if not sentiment_analysis:
            return 'Sentiment Analysis Model not found.'

    text = request.args.get('text')

    result = sentiment_analysis.predict(text)
    return jsonify({
        'status': 'OK',
        'result': result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
