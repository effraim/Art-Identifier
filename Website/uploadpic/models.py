from django.db import models


# Create your models here.
class Artist(models.Model):
    name = models.CharField(max_length=200, default="", unique=True)
    img = models.ImageField(upload_to="images", default="")

    def __str__(self):
        return self.name






