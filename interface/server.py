from flask import Flask
from flask import render_template
from flask import request, jsonify
from flask import url_for
import json
import types
app=Flask(__name__)
from tools import getResults
import ast

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('mofang.html')

@app.route('/initState', methods=['POST'])
def initState():
    if request.method=='POST':
        #rev=request.get_json()['city']
        #result=selcity(rev)
        with open('initState.json', 'r') as f:
            result = json.load(f)
        print(result)
        return jsonify(result)

@app.route('/solve', methods=['POST'])
def solve():
    if request.method == 'POST':
        rev = request.form
        print(rev)
        print("computing...")
        data = rev.to_dict()
        state = []
        data['state'] = ast.literal_eval(data['state'])
        print(data['state'])
        for i in data['state']:
            state.append(int(i))
        result = getResults(state)
        print("complete!")
        return jsonify(result)

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')