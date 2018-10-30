from django.db import models

class Story(models.Model):
    sentences = models.TextField()

    def __str__(self):
        return self.sentences


