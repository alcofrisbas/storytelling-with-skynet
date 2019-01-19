from django.db import models

class Story(models.Model):
    title = models.TextField()
    sentences = models.TextField()

    def __str__(self):
        return self.sentences
