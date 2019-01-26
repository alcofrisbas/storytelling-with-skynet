from django.db import models
from django.contrib.auth.models import AbstractUser

class Story(models.Model):
    title = models.TextField(null=True)
    sentences = models.TextField(null=True)
    public = models.BooleanField(default=False)
    author = models.ForeignKey("User", on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.sentences

class User(AbstractUser):
    email = models.CharField(max_length=255, unique=True, primary_key=True)
    stories = models.ManyToManyField(Story)
    first_name = models.CharField(max_length=255, null=True)
    last_name = models.CharField(max_length=255, null=True)