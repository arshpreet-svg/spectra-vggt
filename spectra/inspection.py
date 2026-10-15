"""Conservative scan advisory driven by external detection events."""
def inspect_events(events, threshold=0.5):
    if not events:
        return {'action':'await_detector','reason':'No external detector observations received',
                'pipegen_execution_verified':False}
    latest=max(events,key=lambda event:(event['timestamp_seconds'],event['frame_index']))
    persons=[item for item in latest['detections'] if item['label'].lower()=='person' and item['confidence']>=threshold]
    return {'action':'pause_scan' if persons else 'clear_observed_frame',
            'reason':'Person detected; wait for a static, unoccupied view before scanning' if persons else 'No person above threshold in the latest observed frame',
            'person_count':len(persons),'frame_index':latest['frame_index'],
            'timestamp_seconds':latest['timestamp_seconds'],'confidence_threshold':threshold,
            'pipegen_execution_verified':False,
            'limitations':'Advisory for the reported frame only; not a safety guarantee or calibrated 3D association'}
