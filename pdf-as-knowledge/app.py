from flask import Flask, request
from ai_engine import qna_with_context

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/qna', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print(request.form)
        return qna_with_context(request.form['context'], request.form['question'])
    return '''
        <form method="post">
            <p><textarea name=context> </textarea>
            <p><input type=text-area name=question>
            <p><input type=submit value=Submit>
        </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)