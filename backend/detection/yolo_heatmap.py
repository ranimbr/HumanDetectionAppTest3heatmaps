import os
import cv2
import numpy as np
import json
import random
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:///mlruns")
model = YOLO("yolov8n.pt")  # modèle YOLOv8 léger

HEATMAP_RADIUS = 20
IMMOBILE_DISTANCE_THRESHOLD = 10  # pixels
FRAME_BLUR_SIZE = 21

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def read_polygons(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    polygons = [np.array(ann["points"], dtype=np.int32) 
                for ann in data["annotations"] if ann["type"]=="polygon"]
    polygons_shapely = [Polygon(p) for p in polygons]
    return polygons, polygons_shapely

def normalize_heatmap(hm):
    return np.uint8(255 * (hm / np.max(hm))**0.7) if np.max(hm) > 0 else np.zeros_like(hm, dtype=np.uint8)

# -----------------------------
# 1. Density Zones
# -----------------------------
def process_video_with_heatmap_density_zones(video_path, json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    polygons, polygons_shapely = read_polygons(json_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, "output_video.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    accumulated_heatmap = np.zeros((height, width), dtype=np.float32)
    history = [[] for _ in range(len(polygons_shapely))]
    total_people_detected = 0
    last_frame_overlay = None

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        person_centers = [(int((x1+x2)/2), int((y1+y2)/2))
                          for x1,y1,x2,y2,score,cls in results.boxes.data.tolist() if int(cls)==0]
        total_people_detected += len(person_centers)

        # Compter par zone
        zone_counts = [sum(1 for x,y in person_centers if poly.contains(Point(x,y))) 
                       for poly in polygons_shapely]

        for i, count in enumerate(zone_counts):
            history[i].append(count)

        # Heatmap cumulée
        for x,y in person_centers:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(accumulated_heatmap, (x,y), HEATMAP_RADIUS, 1.0, -1)

        accumulated_blur = cv2.GaussianBlur(accumulated_heatmap,(FRAME_BLUR_SIZE,FRAME_BLUR_SIZE),sigmaX=15,sigmaY=15)
        accumulated_norm = cv2.normalize(accumulated_blur,None,0,255,cv2.NORM_MINMAX)
        accumulated_colormap = cv2.applyColorMap(accumulated_norm.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame,0.6,accumulated_colormap,0.4,0)
        for i, poly in enumerate(polygons):
            cv2.polylines(overlay,[poly],True,(0,255,0),2)
            cx,cy = polygons_shapely[i].centroid.x, polygons_shapely[i].centroid.y
            cv2.putText(overlay,f"Zone {i+1}: {zone_counts[i]} pers",(int(cx)-50,int(cy)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        out.write(overlay)
        last_frame_overlay = overlay.copy()

    cap.release()
    out.release()

    # Heatmap finale
    mask = np.zeros((height,width),dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask,[poly],255)
    masked_heatmap = cv2.bitwise_and(accumulated_colormap,accumulated_colormap,mask=mask)
    heatmap_path = os.path.join(output_dir,"heatmap.png")
    cv2.imwrite(heatmap_path,masked_heatmap)
    last_frame_path = os.path.join(output_dir,"last_frame.png")
    cv2.imwrite(last_frame_path,last_frame_overlay)

    # Graphique évolution
    evolution_graph_path = os.path.join(output_dir,"zone_counts_evolution.png")
    plt.figure(figsize=(12,6))
    for i, z in enumerate(history):
        plt.plot(z,label=f"Zone {i+1}")
    plt.xlabel("Frame")
    plt.ylabel("Nombre de personnes")
    plt.title("Evolution par zone")
    plt.legend()
    plt.grid(True)
    plt.savefig(evolution_graph_path)
    plt.close()

    return {
        "output_video_path": out_path,
        "heatmap_path": heatmap_path,
        "last_frame_path": last_frame_path,
        "evolution_graph_path": evolution_graph_path,
        "people_count_history": history,
        "total_people_detected": total_people_detected
    }

# -----------------------------
# 2. Stop Time Zones
# -----------------------------
def process_video_with_heatmap_stop_time(video_path, json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    polygons, polygons_shapely = read_polygons(json_path)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_time = 1.0 / fps if fps>0 else 1/30

    out_path = os.path.join(output_dir,"output_video.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    track_positions = {}
    track_stop_times = {}
    zone_stop_times = np.zeros((height,width),dtype=np.float32)
    zone_total_stop_times = [0]*len(polygons_shapely)
    zone_stop_history = [[] for _ in range(len(polygons_shapely))]
    last_frame_overlay = None

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for box in results.boxes.data.tolist():
            x1,y1,x2,y2,score,cls = box
            if int(cls)==0:
                detections.append(([x1,y1,x2-x1,y2-y1],score,'person'))

        tracks = tracker.update_tracks(detections,frame=frame)
        frame_zone_stops = [0]*len(polygons_shapely)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l,t,r,b = track.to_ltrb()
            cx,cy = int((l+r)/2),int((t+b)/2)
            point = Point(cx,cy)

            prev_pos = track_positions.get(track_id,(cx,cy))
            dist = np.linalg.norm(np.array([cx,cy])-np.array(prev_pos))
            track_positions[track_id] = (cx,cy)

            if dist<IMMOBILE_DISTANCE_THRESHOLD:
                zone_stop_times[cy,cx] += frame_time
                track_stop_times[track_id] = track_stop_times.get(track_id,0)+frame_time
                for i,poly in enumerate(polygons_shapely):
                    if poly.contains(point):
                        zone_total_stop_times[i] += frame_time
                        frame_zone_stops[i] += frame_time
                        break

            label = f"ID:{track_id} | {track_stop_times.get(track_id,0):.1f}s"
            cv2.rectangle(frame,(int(l),int(t)),(int(r),int(b)),(0,255,0),2)
            cv2.putText(frame,label,(int(l),int(t)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        for i in range(len(polygons_shapely)):
            zone_stop_history[i].append(frame_zone_stops[i])

        heatmap_blur = cv2.GaussianBlur(zone_stop_times,(51,51),0)
        heatmap_norm = cv2.normalize(heatmap_blur,None,0,255,cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame,0.6,heatmap_color,0.4,0)
        for i,poly in enumerate(polygons):
            cv2.polylines(overlay,[poly],True,(0,255,0),2)
            cx,cy = polygons_shapely[i].centroid.x, polygons_shapely[i].centroid.y
            cv2.putText(overlay,f"Zone {i+1}: {zone_total_stop_times[i]:.1f}s",(int(cx)-50,int(cy)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        out.write(overlay)
        last_frame_overlay = overlay.copy()

    cap.release()
    out.release()

    # Heatmap finale
    mask = np.zeros((height,width),dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask,[poly],255)
    masked_heatmap = cv2.bitwise_and(heatmap_color,heatmap_color,mask=mask)
    heatmap_path = os.path.join(output_dir,"heatmap.png")
    cv2.imwrite(heatmap_path,masked_heatmap)
    last_frame_path = os.path.join(output_dir,"last_frame.png")
    cv2.imwrite(last_frame_path,last_frame_overlay)

    # Graphiques
    evolution_graph_path = os.path.join(output_dir,"stop_history.png")
    plt.figure(figsize=(15,8))
    zones = [f"Zone {i+1}" for i in range(len(polygons_shapely))]
    plt.subplot(1,2,1)
    plt.bar(zones,zone_total_stop_times,color='skyblue')
    plt.title("Temps d'arrêt total par zone")
    plt.subplot(1,2,2)
    for i, history in enumerate(zone_stop_history):
        smoothed = pd.Series(history).rolling(window=30,min_periods=1).mean()
        plt.plot(smoothed,label=f"Zone {i+1}")
    plt.title("Évolution temps d'arrêt par zone")
    plt.tight_layout()
    plt.savefig(evolution_graph_path)
    plt.close()

    return {
        "output_video_path": out_path,
        "heatmap_path": heatmap_path,
        "last_frame_path": last_frame_path,
        "evolution_graph_path": evolution_graph_path,
        "zone_stop_history": zone_stop_history
    }

# -----------------------------
# 3. Product Interaction
# -----------------------------
def process_video_with_heatmap_product_interaction(video_path, json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    polygons, polygons_shapely = read_polygons(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Vidéo annotée en sortie
    out_path = os.path.join(output_dir, "output_video.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    accumulated_heatmap_product = np.zeros((height, width), dtype=np.float32)  # Heatmap globale
    product_heatmap = np.zeros((height, width), dtype=np.float32)  # Heatmap centrée sur les produits
    last_frame_overlay = None
    interaction_history = [[] for _ in polygons_shapely]

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        # Détection des personnes uniquement
        person_centers = [(int((x1+x2)/2), int((y1+y2)/2))
                          for x1, y1, x2, y2, score, cls in results.boxes.data.tolist()
                          if int(cls) == 0]

        # Vérifier si la personne est dans une zone produit
        for x, y in person_centers:
            point = Point(x, y)
            for i, poly in enumerate(polygons_shapely):
                if poly.contains(point):
                    # Ajouter chaleur dans la heatmap produit
                    cv2.circle(accumulated_heatmap_product, (x, y), HEATMAP_RADIUS, 1.0, -1)
                    cv2.circle(product_heatmap, (x, y), HEATMAP_RADIUS, 1.0, -1)
                    interaction_history[i].append(1)
                else:
                    interaction_history[i].append(0)

        # Générer heatmap cumulative
        blur_product = cv2.GaussianBlur(accumulated_heatmap_product, (FRAME_BLUR_SIZE, FRAME_BLUR_SIZE), 0)
        norm_product = cv2.normalize(blur_product, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(norm_product.astype(np.uint8), cv2.COLORMAP_JET)

        # Générer heatmap spécifique produits
        blur_prod_specific = cv2.GaussianBlur(product_heatmap, (FRAME_BLUR_SIZE, FRAME_BLUR_SIZE), 0)
        norm_prod_specific = cv2.normalize(blur_prod_specific, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_product_color = cv2.applyColorMap(norm_prod_specific.astype(np.uint8), cv2.COLORMAP_JET)

        # Fusionner avec la frame originale
        overlay = cv2.addWeighted(frame, 0.6, heatmap_product_color, 0.4, 0)

        # Dessiner zones et compteur
        for i, poly in enumerate(polygons):
            cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)
            cx, cy = polygons_shapely[i].centroid.x, polygons_shapely[i].centroid.y
            cv2.putText(
                overlay,
                f"Zone {i+1}: {sum(interaction_history[i])} interactions",
                (int(cx)-50, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        last_frame_overlay = overlay.copy()
        out.write(overlay)

    cap.release()
    out.release()

    # Sauvegarde des images finales
    heatmap_path = os.path.join(output_dir, "heatmap_global.png")
    cv2.imwrite(heatmap_path, heatmap_color)

    heatmap_product_path = os.path.join(output_dir, "heatmap_products.png")
    cv2.imwrite(heatmap_product_path, heatmap_product_color)

    last_frame_path = os.path.join(output_dir, "last_frame.png")
    cv2.imwrite(last_frame_path, last_frame_overlay)

    # Graphique évolution interactions
    evolution_graph_path = os.path.join(output_dir, "interaction_evolution.png")
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(interaction_history):
        plt.plot(np.cumsum(history), label=f"Zone {i+1}")
    plt.xlabel("Frames")
    plt.ylabel("Nombre cumulé d'interactions")
    plt.title("Évolution des interactions par zone produit")
    plt.legend()
    plt.grid(True)
    plt.savefig(evolution_graph_path)
    plt.close()

    return {
    "output_video_path": out_path,
    "heatmap_path": heatmap_path,   
    "heatmap_products_path": heatmap_product_path,
    "last_frame_path": last_frame_path,
    "evolution_graph_path": evolution_graph_path,
    "interaction_history": interaction_history
}


# -----------------------------
# 4. Image Detection
# -----------------------------
def process_image_with_heatmap(image_path, json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    polygons, polygons_shapely = read_polygons(json_path)

    img = cv2.imread(image_path)
    results = model(img)[0]
    person_centers = [(int((x1+x2)/2), int((y1+y2)/2))
                      for x1,y1,x2,y2,score,cls in results.boxes.data.tolist() if int(cls)==0]

    zone_counts = [sum(1 for x,y in person_centers if poly.contains(Point(x,y))) 
                   for poly in polygons_shapely]
    total_people_detected = len(person_centers)

    accumulated_heatmap = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    for x,y in person_centers:
        cv2.circle(accumulated_heatmap,(x,y),HEATMAP_RADIUS,1.0,-1)
    accumulated_blur = cv2.GaussianBlur(accumulated_heatmap,(FRAME_BLUR_SIZE,FRAME_BLUR_SIZE),0)
    accumulated_norm = cv2.normalize(accumulated_blur,None,0,255,cv2.NORM_MINMAX)
    accumulated_colormap = cv2.applyColorMap(accumulated_norm.astype(np.uint8),cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,accumulated_colormap,0.4,0)
    for i, poly in enumerate(polygons):
        cv2.polylines(overlay,[poly],True,(0,255,0),2)
        cx,cy = polygons_shapely[i].centroid.x, polygons_shapely[i].centroid.y
        cv2.putText(overlay,f"Zone {i+1}: {zone_counts[i]} pers",(int(cx)-50,int(cy)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    annotated_path = os.path.join(output_dir,"annotated_image.png")
    heatmap_path = os.path.join(output_dir,"heatmap.png")
    cv2.imwrite(annotated_path,overlay)
    cv2.imwrite(heatmap_path,accumulated_colormap)

    return {
        "annotated_image_path": annotated_path,
        "heatmap_path": heatmap_path,
        "zone_counts": zone_counts,
        "total_people_detected": total_people_detected
    }
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["density", "stop", "product", "image"])
    parser.add_argument("--video", type=str, help="Chemin de la vidéo")
    parser.add_argument("--json", type=str, help="Chemin du fichier JSON zones")
    parser.add_argument("--image", type=str, help="Chemin de l'image (mode image uniquement)")
    parser.add_argument("--output", type=str, required=True, help="Dossier de sortie")

    args = parser.parse_args()

    if args.mode == "density":
        process_video_with_heatmap_density_zones(args.video, args.json, args.output)

    elif args.mode == "stop":
        process_video_with_heatmap_stop_time(args.video, args.json, args.output)

    elif args.mode == "product":
        process_video_with_heatmap_product_interaction(args.video, args.json, args.output)

    elif args.mode == "image":
        process_image_with_heatmap(args.image, args.json, args.output)

