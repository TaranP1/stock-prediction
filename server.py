from flask import Flask, render_template, request
from stockInfo import lstmPredict
from waitress import serve
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/stock', methods=['GET'])
def get_data():
    symbol = request.args.get('ticker').upper()
    stock_data = lstmPredict(symbol)
    return render_template(
        "stock.html",
        ticker = symbol
    )
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)