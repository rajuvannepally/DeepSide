from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,drug_side_effect_prediction,detection_ratio,detection_accuracy
from sklearn.tree import DecisionTreeClassifier
def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":

            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            phoneno = request.POST.get('phoneno')
            country = request.POST.get('country')
            state = request.POST.get('state')
            city = request.POST.get('city')
            address = request.POST.get('address')
            gender = request.POST.get('gender')
            ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                                country=country, state=state, city=city, address=address, gender=gender)
            obj = "Registered Successfully"
            return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Drug_Side_Effect_Type(request):
    if request.method == "POST":

        uid= request.POST.get('uid')
        Drug_Name= request.POST.get('Drug_Name')
        Condition1= request.POST.get('Condition1')


        df = pd.read_csv('Datasets.csv',encoding='latin-1')
        df
        df.columns

        def apply_results(Rating):
            if (int(Rating) <= 7):
                return 0  # Low Side Effect
            else:
                return 1  # High Side Effect

        df['Results'] = df['Rating'].apply(apply_results)

        cv = CountVectorizer()
        X = df['UID'].apply(str)
        y = df['Results']

        print("UID")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()

        X = cv.fit_transform(X)
        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Multi-modal neural networks (MMNN)")

        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))


        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))


        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        uid1 = [uid]
        vector1 = cv.transform(uid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Low Side Effect Found'
        elif prediction == 1:
            val = 'High Side Effect Found'

        print(val)
        print(pred1)

        drug_side_effect_prediction.objects.create(
        uid=uid,
        Drug_Name=Drug_Name,
        Condition1=Condition1,

        Prediction=val)

        return render(request, 'RUser/Predict_Drug_Side_Effect_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Drug_Side_Effect_Type.html')



