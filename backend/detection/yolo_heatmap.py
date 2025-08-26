import os
import cv2
import numpy as np
import json
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
mlflow.set_tracking_uri("file:///mlruns")
def process_image_with_heatmap(image_path, json_path, output_dir):
    mlflow.set_experiment("HumanDetectionImage")
    with mlflow.start_run():

        os.makedirs(output_dir, exist_ok=True)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        polygons = []
        for ann in data['annotations']:
            if ann['type'] == 'polygon':
                pts = np.array(ann['points'], dtype=np.int32)
                polygons.append(pts)
        polygons_shapely = [Polygon(poly) for poly in polygons]

        frame = cv2.imread(image_path)
        if frame is None:
            raise IOError(f"Impossible de charger l'image {image_path}")

        height, width = frame.shape[:2]
        model = YOLO('yolov8n.pt')
        results = model(frame)[0]

        person_centers = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            if int(cls) == 0:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                person_centers.append((cx, cy))

        zone_counts = [0] * len(polygons_shapely)
        zone_centers = []

        for i, poly in enumerate(polygons_shapely):
            count = 0
            for x, y in person_centers:
                pt = Point(x, y)
                if poly.contains(pt):
                    count += 1
            zone_counts[i] = count
            centroid = poly.centroid
            zone_centers.append((int(centroid.x), int(centroid.y)))

        heatmap_frame = np.zeros((height, width), dtype=np.float32)
        for x, y in person_centers:
            if 0 <= y < height and 0 <= x < width:
                cv2.circle(heatmap_frame, (x, y), 20, 1.0, thickness=-1)

        heatmap_blur = cv2.GaussianBlur(heatmap_frame, (0, 0), sigmaX=15, sigmaY=15)
        heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)

        for i, poly in enumerate(polygons):
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            text = f"Zone {i+1}: {zone_counts[i]} pers"
            cx, cy = zone_centers[i]
            cv2.putText(overlay, text, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        annotated_image_path = os.path.join(output_dir, "annotated_image.png")
        heatmap_path = os.path.join(output_dir, "heatmap.png")
        cv2.imwrite(annotated_image_path, overlay)
        cv2.imwrite(heatmap_path, colored_heatmap)

        # MLflow logs
        mlflow.log_param("image_path", image_path)
        mlflow.log_metric("total_people_detected", len(person_centers))
        for i, count in enumerate(zone_counts):
            mlflow.log_metric(f"zone_{i+1}_count", count)
        mlflow.log_artifact(annotated_image_path)
        mlflow.log_artifact(heatmap_path)

        return {
            "annotated_image_path": annotated_image_path,
            "heatmap_path": heatmap_path,
            "zone_counts": zone_counts,
            "total_people_detected": len(person_centers)
        }


def _process_video_with_heatmap_generic(video_path, json_path, output_dir):
    mlflow.set_experiment("HumanDetectionVideo")
    with mlflow.start_run():

        os.makedirs(output_dir, exist_ok=True)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        polygons = []
        for ann in data['annotations']:
            if ann['type'] == 'polygon':
                pts = np.array(ann['points'], dtype=np.int32)
                polygons.append(pts)
        polygons_shapely = [Polygon(poly) for poly in polygons]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Impossible d'ouvrir la vidéo")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = os.path.join(output_dir, "output_with_heatmap.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        model = YOLO('yolov8n.pt')
        history = [[] for _ in range(len(polygons_shapely))]
        accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

        total_people_detected = 0
        last_overlay_frame = None

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            person_centers = [
                (int((x1 + x2) / 2), int((y1 + y2) / 2))
                for x1, y1, x2, y2, score, cls in results.boxes.data.tolist()
                if int(cls) == 0
            ]

            total_people_detected += len(person_centers)

            zone_counts = [0] * len(polygons_shapely)
            zone_centers = []

            for i, poly in enumerate(polygons_shapely):
                count = sum(1 for x, y in person_centers if poly.contains(Point(x, y)))
                zone_counts[i] = count
                centroid = poly.centroid
                zone_centers.append((int(centroid.x), int(centroid.y)))
                history[i].append(count)

            for x, y in person_centers:
                if 0 <= y < frame_height and 0 <= x < frame_width:
                    cv2.circle(accumulated_heatmap, (x, y), 20, 1.0, thickness=-1)

            accumulated_blur = cv2.GaussianBlur(accumulated_heatmap, (21, 21), sigmaX=15, sigmaY=15)
            accumulated_norm = cv2.normalize(accumulated_blur, None, 0, 255, cv2.NORM_MINMAX)
            accumulated_uint8 = accumulated_norm.astype(np.uint8)
            accumulated_colormap = cv2.applyColorMap(accumulated_uint8, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(frame, 0.6, accumulated_colormap, 0.4, 0)
            for i, poly in enumerate(polygons):
                cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
                text = f"Zone {i+1}: {zone_counts[i]} pers"
                cx, cy = zone_centers[i]
                cv2.putText(overlay, text, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(overlay)
            last_overlay_frame = overlay.copy()

        cap.release()
        out.release()

        graph_path = os.path.join(output_dir, "zone_counts_evolution.png")
        plt.figure(figsize=(12, 6))
        for i, zone_history in enumerate(history):
            plt.plot(zone_history, label=f"Zone {i+1}")
        plt.title("Évolution du nombre de personnes par zone")
        plt.xlabel("Frame")
        plt.ylabel("Nombre de personnes")
        plt.legend()
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()

        last_frame_path = os.path.join(output_dir, "last_frame_with_heatmap.png")
        if last_overlay_frame is not None:
            cv2.imwrite(last_frame_path, last_overlay_frame)

        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 255)
        masked_heatmap = cv2.bitwise_and(accumulated_colormap, accumulated_colormap, mask=mask)
        heatmap_path = os.path.join(output_dir, "heatmap_cumulative_masked.png")
        cv2.imwrite(heatmap_path, masked_heatmap)

        # MLflow logs
        mlflow.log_param("video_path", video_path)
        mlflow.log_param("frame_count", frame_count)
        mlflow.log_metric("total_people_detected", total_people_detected)
        mlflow.log_artifact(output_video_path)
        mlflow.log_artifact(heatmap_path)
        mlflow.log_artifact(last_frame_path)
        mlflow.log_artifact(graph_path)

        return {
            "output_video_path": output_video_path,
            "heatmap_path": heatmap_path,
            "last_frame_path": last_frame_path,
            "evolution_graph_path": graph_path,
            "people_count_history": history,
            "total_people_detected": total_people_detected
        }


def process_video_with_heatmap_density_zones(video_path, json_path, output_dir):
    return _process_video_with_heatmap_generic(video_path, json_path, output_dir)


def process_video_with_heatmap_stop_time(video_path, json_path, output_dir):
    return _process_video_with_heatmap_generic(video_path, json_path, output_dir)


def process_video_with_heatmap_product_interaction(video_path, json_path, output_dir):
    return _process_video_with_heatmap_generic(video_path, json_path, output_dir)


