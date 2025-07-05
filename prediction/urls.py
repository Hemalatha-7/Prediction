from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.loginpage,name='login'),
    path('home/',views.home,name='home'),
    path('register/',views.registerpage,name='register'),
    path('ml/',views.ml_predict,name='ml'),
    path('dl/',views.dl_predict,name='dl'),
    path('logout/',views.logoutpage,name='logout')
]
