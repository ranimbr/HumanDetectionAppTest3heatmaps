import mongoengine as me
from datetime import datetime

class DetectionResult(me.Document):
    video_name = me.StringField(required=True)
    zones = me.ListField(me.StringField())
    people_count_history = me.ListField(me.ListField(me.IntField()))
    heatmap_path = me.StringField()
    last_frame_path = me.StringField()
    evolution_graph_path = me.StringField()
    timestamp = me.DateTimeField(default=datetime.utcnow)
