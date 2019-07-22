from flask import Flask, request, jsonify
from sentiment_analysis import SentimentAnalysis
from conversation import Conversation

SA_TOKENIZER = './models/tokenizer.pkl'
SA_MODEL = './models/model.h5'
CONVERSATION_MODEL = './models/s2s.h5'

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global sentiment_analysis
    global conversation

    sentiment_analysis = SentimentAnalysis(SA_TOKENIZER, SA_MODEL)
    conversation = Conversation(CONVERSATION_MODEL)


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
    # result example
    # {'label': 'NEGATIVE',
    # 'score': 0.010753681883215904,
    # 'elapsed_time': 0.26644086837768555}
    return jsonify({
        'status': 'OK',
        'result': result
    })


@app.route('/talk')
def talk():
    if not conversation:
        _load_model()
        if not conversation:
            return 'Conversation Model not found.'

    text = request.args.get('text')

    result = conversation.reply(text)
    # result example
    # {'input_seq': 'how are you',
    # 'output_seq': 'i am fine'}
    return jsonify({
        'status': 'OK',
        'result': result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
