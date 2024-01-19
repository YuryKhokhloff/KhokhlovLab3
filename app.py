from flask import Flask, request,render_template
import pickle

app=Flask(__name__, template_folder='templates')
model=pickle.load(open('modellab2.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.get("input_data")
    age = request.form['age']
    edu = request.form['edu']

    age = float(age)
    edu = float(edu)

    prediction = model.predict([[age, edu]])

    return render_template('result.html', prediction=prediction)


if __name__=="__main__":
    app.run(port=5000, debug=True)