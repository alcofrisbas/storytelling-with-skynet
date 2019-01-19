from django.db import models

class Story(models.Model):
    title = models.TextField(null=True)
    sentences = models.TextField(null=True)

    def __str__(self):
        return self.sentences
