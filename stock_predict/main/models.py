from django.db import models

class News(models.Model):
    title = models.CharField(max_length=150)
    date = models.CharField(max_length=50)

    class Meta:
        db_table = 'news'
