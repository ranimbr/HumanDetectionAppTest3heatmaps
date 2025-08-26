import pytest
import os
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

FILES_MAPPING = {
    'density': {
        'video': 'tests_files/test_video_density.mp4',
        'json': 'tests_files/test_zones_density.json',
    },
    'stop_time': {
        'video': 'tests_files/test_video_stop.mp4',
        'json': 'tests_files/test_zones_stop.json',
    },
    'product_interaction': {
        'video': 'tests_files/test_video_interaction.mp4',
        'json': 'tests_files/test_zones_interaction.json',
    }
}

def get_uploaded_files(test_key):
    paths = FILES_MAPPING[test_key]
    if not (os.path.exists(paths['video']) and os.path.exists(paths['json'])):
        pytest.skip(f"Fichiers de test manquants pour {test_key}.")
    
    with open(paths['video'], 'rb') as vf, open(paths['json'], 'rb') as jf:
        video = SimpleUploadedFile(os.path.basename(paths['video']), vf.read(), content_type="video/mp4")
        json_file = SimpleUploadedFile(os.path.basename(paths['json']), jf.read(), content_type="application/json")
        return video, json_file

@pytest.mark.django_db
def test_index_view(client):
    url = reverse('index')
    response = client.get(url)
    assert response.status_code == 200
    assert "YOLO Video Heatmap Detection" in response.content.decode('utf-8')

@pytest.mark.django_db
def test_upload_video_density_zones_with_files(client):
    video, json_file = get_uploaded_files('density')
    response = client.post(
        reverse('upload_video_density_zones'),
        data={
            'video_file': video,
            'json_file': json_file,
        }
    )
    assert response.status_code == 200
    assert "Detection Results" in response.content.decode('utf-8')

@pytest.mark.django_db
def test_upload_video_stop_time_with_files(client):
    video, json_file = get_uploaded_files('stop_time')
    response = client.post(
        reverse('upload_video_stop_time'),
        data={
            'video_file': video,
            'json_file': json_file,
        }
    )
    assert response.status_code == 200
    assert "Detection Results" in response.content.decode('utf-8')

@pytest.mark.django_db
def test_upload_video_product_interaction_with_files(client):
    video, json_file = get_uploaded_files('product_interaction')
    response = client.post(
        reverse('upload_video_product_interaction'),
        data={
            'video_file': video,
            'json_file': json_file,
        }
    )
    assert response.status_code == 200
    assert "Detection Results" in response.content.decode('utf-8')
