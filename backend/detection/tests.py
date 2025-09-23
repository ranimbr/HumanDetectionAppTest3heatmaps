import pytest
import os
import tempfile
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

# Pour créer une vidéo temporaire valide
from moviepy.editor import ColorClip

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

    # Générer une vidéo temporaire si le fichier n'existe pas
    if not os.path.exists(paths['video']):
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        clip = ColorClip(size=(640, 480), color=(0,0,0), duration=1)  # 1 sec, noir
        clip.write_videofile(temp_video.name, fps=24, codec='libx264', audio=False, verbose=False, logger=None)
        video_path = temp_video.name
    else:
        video_path = paths['video']

    # Générer un JSON temporaire si le fichier n'existe pas
    if not os.path.exists(paths['json']):
        json_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        with open(json_path, 'w') as f:
            f.write("{}")
    else:
        json_path = paths['json']

    with open(video_path, 'rb') as vf, open(json_path, 'rb') as jf:
        video = SimpleUploadedFile(os.path.basename(video_path), vf.read(), content_type="video/mp4")
        json_file = SimpleUploadedFile(os.path.basename(json_path), jf.read(), content_type="application/json")
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
