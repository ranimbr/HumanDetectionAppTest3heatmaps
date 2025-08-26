import os
import base64
import cv2
import numpy as np
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from datetime import datetime

from .models import DetectionResult
from .yolo_heatmap import (
    process_video_with_heatmap_density_zones,  # à créer
    process_video_with_heatmap_stop_time,      # à créer
    process_video_with_heatmap_product_interaction,  # à créer
    process_image_with_heatmap
)

from ultralytics import YOLO

# Charger le modèle YOLO globalement pour éviter de le charger à chaque requête
yolo_model = YOLO('yolov8n.pt')

def index(request):
    return render(request, 'index.html')

def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        json_file = request.FILES.get('json')

        if not image or not json_file:
            return HttpResponse("Image or JSON file missing", status=400)

        uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"{timestamp}_{image.name}"
        json_filename = f"{timestamp}_{json_file.name}"

        image_path = os.path.join(uploads_dir, image_filename)
        json_path = os.path.join(uploads_dir, json_filename)

        try:
            with open(image_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            with open(json_path, 'wb+') as f:
                for chunk in json_file.chunks():
                    f.write(chunk)

            output_dir = os.path.join(settings.MEDIA_ROOT, 'results', timestamp)
            os.makedirs(output_dir, exist_ok=True)

            results = process_image_with_heatmap(image_path, json_path, output_dir)

        except Exception as e:
            return HttpResponse(f"Error processing image: {e}", status=500)

        rel_annotated_path = os.path.relpath(results["annotated_image_path"], settings.MEDIA_ROOT).replace('\\', '/')
        rel_heatmap_path = os.path.relpath(results["heatmap_path"], settings.MEDIA_ROOT).replace('\\', '/')

        context = {
            'input_type': 'image',
            'annotated_image_path': rel_annotated_path,
            'heatmap_path': rel_heatmap_path,
            'zone_counts': results.get("zone_counts", []),
            'total_people_detected': results.get("total_people_detected", 0),
            'original_filename': image.name,
            'detection_details': f"Personnes détectées: {results.get('total_people_detected', 0)} par zone: {results.get('zone_counts', [])}"
        }
        return render(request, 'result.html', context)

    return redirect('index')

def detect_video1(request):  # Détection "density zones"
    return _process_video_detection(request, process_video_with_heatmap_density_zones, "Density Zones")

def detect_video2(request):  # Détection "stop time zones"
    return _process_video_detection(request, process_video_with_heatmap_stop_time, "Stop Time Zones")

def detect_video3(request):  # Détection "product interaction"
    return _process_video_detection(request, process_video_with_heatmap_product_interaction, "Product Interaction")

def _process_video_detection(request, process_function, detection_name):
    if request.method == 'POST':
        video = request.FILES.get('video_file')
        json_file = request.FILES.get('json_file')

        if not video or not json_file:
            return HttpResponse("Video or JSON file missing", status=400)

        uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"{timestamp}_{video.name}"
        json_filename = f"{timestamp}_{json_file.name}"

        video_path = os.path.join(uploads_dir, video_filename)
        json_path = os.path.join(uploads_dir, json_filename)

        try:
            with open(video_path, 'wb+') as vf:
                for chunk in video.chunks():
                    vf.write(chunk)

            with open(json_path, 'wb+') as jf:
                for chunk in json_file.chunks():
                    jf.write(chunk)

            output_dir = os.path.join(settings.MEDIA_ROOT, 'results', timestamp)
            os.makedirs(output_dir, exist_ok=True)

            results = process_function(video_path, json_path, output_dir)

        except Exception as e:
            return HttpResponse(f"Error processing video ({detection_name}): {e}", status=500)

        rel_output_video_path = os.path.relpath(results["output_video_path"], settings.MEDIA_ROOT).replace('\\', '/')
        rel_heatmap_path = os.path.relpath(results["heatmap_path"], settings.MEDIA_ROOT).replace('\\', '/')
        rel_last_frame_path = os.path.relpath(results["last_frame_path"], settings.MEDIA_ROOT).replace('\\', '/')
        rel_evolution_path = os.path.relpath(results["evolution_graph_path"], settings.MEDIA_ROOT).replace('\\', '/')

        # Optionnel : sauvegarde en base
        DetectionResult.objects.create(
            video_name=video.name,
            zones=[],
            people_count_history=results.get("people_count_history", []),
            heatmap_path=rel_heatmap_path,
            last_frame_path=rel_last_frame_path,
            evolution_graph_path=rel_evolution_path,
            timestamp=datetime.utcnow()
        )

        context = {
            'input_type': 'video',
            'output_video_path': rel_output_video_path,
            'heatmap_path': rel_heatmap_path,
            'last_frame_path': rel_last_frame_path,
            'evolution_graph_path': rel_evolution_path,
            'original_filename': video.name,
            'detection_details': f"Résultats de détection ({detection_name}) pour {video.name}",
        }

        return render(request, 'result.html', context)

    return redirect('index')

