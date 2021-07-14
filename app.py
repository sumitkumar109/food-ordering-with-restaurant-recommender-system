from flask import Flask, render_template, request
from recommendation import restaurant_recommendation

app = Flask(__name__)

@app.route('/')
def load():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    output = restaurant_recommendation(request.form.get('locality'), request.form.get('title'))
    names = []
    cuisines = []
    ratings = []
    costs = []
    for i in range(len(output)):
        names.append(output.iloc[i][0])
        ratings.append(output.iloc[i][1])
        cuisines.append(output.iloc[i][2])
        costs.append(output.iloc[i][3])
    return render_template('recommendation.html', name=names, rating=ratings, cuisine=cuisines, cost=costs)


if __name__ == "__main__":
    app.run(debug=True)