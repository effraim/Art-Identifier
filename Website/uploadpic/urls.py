from django.urls import path, include
from. import views
from django.conf import settings
from django.conf.urls.static import static
from .views import image_upload_view
from .views import results

app_name = 'uploadpic'

urlpatterns = [
    path('', views.index, name='index'),
    #path('', image_upload_view, name="image_upload_view"),
    path('uploadpic/', views.image_upload_view, name="upload_pic"),
    path('results/', views.results, name="results"),
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_URL)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

