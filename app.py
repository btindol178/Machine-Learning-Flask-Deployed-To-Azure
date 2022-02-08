from flask import Flask, render_template,redirect,request,url_for
import predict
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route("/",methods = ['POST','GET'])
def index():
    try:
        if request.method == 'POST':
            pregnancies = request.form.get("pregnancies")
            glucose = request.form.get("glucose")
            bloodpressure = request.form.get("bloodpressure")
            skinthickness = request.form.get("skinthickness")
            insulin = request.form.get("insulin")
            bmi = request.form.get("bmi")
            dpfunc = request.form.get("dpfunc")
            age = request.form.get("age")
            model = request.form.get("model")
            print(pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,dpfunc,age,model)

            # Put these variables in the same order as you did for predict.py 
            result = predict.predict(pregnancies,
                glucose,
                bloodpressure,
                skinthickness,
                insulin,
                bmi,
                dpfunc,
                age,
                model)
            
            if result == 1:
                message = "It seems you have diabetes, talk to your doctor!"
            else:
                message = "It seems you are healthy! Eat healthy stay healthy!!"
            print("The predicted result is:",result)
            print(message)
            return render_template("index.html",data = [{"msg":message}])
    except:
        return "Please check if values are entered correctly"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)