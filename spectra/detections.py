"""Boundary contract for a separately generated PipeGen detector.

This validates events; receiving an event does not prove PipeGen execution.
Boxes use pixels in the original detector image, not VGGT-cropped images.
"""
import math

def validate_event(event):
    if not isinstance(event, dict) or event.get('schema_version') != 1:
        raise ValueError('Expected schema_version=1 object')
    if event.get('producer') != 'pipegen':
        raise ValueError('producer must be pipegen')
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    frame = event.get('frame_index')
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError('frame_index must be a nonnegative integer')
    timestamp = event.get('timestamp_seconds')
    if not number(timestamp) or timestamp < 0:
        raise ValueError('timestamp_seconds must be finite and nonnegative')
    size = event.get('image_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError('image_size must be [width,height] positive integers')
    detections = event.get('detections')
    if not isinstance(detections, list) or len(detections) > 1000:
        raise ValueError('detections must be a list with at most 1000 entries')
    for item in detections:
        if not isinstance(item, dict) or not isinstance(item.get('label'), str) or not item['label'].strip():
            raise ValueError('Each detection needs a nonempty label')
        score, box = item.get('confidence'), item.get('bbox_xyxy')
        if not number(score) or not 0 <= score <= 1:
            raise ValueError('confidence must be between 0 and 1')
        if not isinstance(box, list) or len(box) != 4 or not all(number(x) for x in box):
            raise ValueError('bbox_xyxy must contain four finite numbers')
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise ValueError('bbox_xyxy must fit image_size')
    return {key: event[key] for key in ['schema_version', 'producer', 'frame_index', 'timestamp_seconds', 'image_size', 'detections']}
