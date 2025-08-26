from django import forms

class UploadVideoForm(forms.Form):
    video = forms.FileField(label="Upload a video")
