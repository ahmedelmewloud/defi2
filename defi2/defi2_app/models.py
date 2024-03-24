from django.db import models

class Mougataa(models.Model):
    nom = models.CharField(max_length=100)
    wilaya = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()


    def __str__(self):
        return self.nom
class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploaded_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.nom