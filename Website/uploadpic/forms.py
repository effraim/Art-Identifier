# import form class from django
from django import forms

from django.db import models  #now
from django.forms import fields  #now

# import Artist from models.py
from .models import Artist


# create a ModelForm
class ArtistForm(forms.ModelForm):
    # specify the name of model to use
    class Meta:
        model = Artist
        fields = '__all__'


class QueryForm(forms.Form):
    item = forms.ModelChoiceField(
        widget=forms.Select,
        queryset=Artist.objects.all()
    )
