from django.shortcuts import render
from django.http import HttpResponse
from .models import Artist
from django.shortcuts import render, redirect
from .forms import ArtistForm, QueryForm
from .process_image.test_an_image import results_pic
# Create your views here.


def index(request):
    return render(request, 'home.html')


def results(request):
    if request.method == 'POST':
        form = ArtistForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_object = form.instance
            res = results_pic(img_object.img)
            return render(request, 'results.html', {'img_obj': img_object, 'res': res})
    if request.method == 'GET':
        if len(request.GET) > 0:
            form = QueryForm(request.GET)
            if form.is_valid():
                img_object = Artist.objects.filter(id=int(request.GET['item']))[0]
                img = getattr(img_object, 'img')
                res = results_pic(img)
                return render(request, 'results.html', {'img': img, 'res': res})
            else:
                print(form)
    form = QueryForm()
    return render(request, 'results.html', {'form': form})

# def index2(request):
#     return render(request, 'uploadpic.html')


def image_upload_view(request):
    """Process images uploaded by users"""
    form = ArtistForm()
    return render(request, 'uploadpic.html', {'form': form})
