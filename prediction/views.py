from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import io

# Create your views here.
def loginpage(request):
    if request.method=='POST':
        username=request.POST.get('uname')
        password=request.POST.get('pword')
        user=authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('home')
    return render(request, 'login.html')
def registerpage(request):
    if request.method=='POST':
        username=request.POST.get('uname')
        password=request.POST.get('pword')
        conform=request.POST.get('pword2')
        if password!=conform:
            return render(request, 'register.html', {'result': 'Passwords do not match'})
        User.objects.create_user(username=username,password=password)
        return redirect('login')
    return render(request, 'register.html')

def dl_predict(request):
    model1 = load_model('sic_model.h5')
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        img_bytes=img_file.read()
        img_io = io.BytesIO(img_bytes)
        img = image.load_img(img_io, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  
        prediction = model1.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = predicted_class
        if result == 0:
            result = "Cloudy"
        elif result == 1:
            result = "Desert"   
        elif result == 2:
            result = "Green Area"
        else:
            result = "Water"
            
    return render(request, 'dl.html', {'result': result})
def ml_predict(request):
    result = None
    model_file='sme_model.pkl'
    model=pickle.load(open(model_file, 'rb'))
    if request.method == 'POST':
        a = int(request.POST.get('platform'))
        b = int(request.POST.get('likes'))
        c = int(request.POST.get('comments'))
        d = int(request.POST.get('shares'))
        p = model.predict([[a, b, c, d]])
        result = p[0]
        if result == 0:
            result = "Negative sentiment"
        elif result == 1:
            result = "Neutral sentiment"
        else:
            result = "Positive sentiment"
    return render(request, 'ml.html', {'result': result})

def home(request):
    if request.method == 'POST':
        action= request.POST.get('action')
        if action == 'mlearning':
            return redirect('ml')  
        elif action == 'dlearning':
            return redirect('dl')  
    return render(request, 'home.html')

def logoutpage(request):
    return redirect('login')

