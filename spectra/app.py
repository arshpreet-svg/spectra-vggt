import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from .config import Config
from .detections import validate_event
from .engine import make_engine
from .inputs import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .inspection import inspect_events
from .reconstruction import reconstruct


def create_app(config=None, engine=None):
    config = config or Config.from_env()
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = config.max_upload_mb * 1024 * 1024
    config.output_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or make_engine(config)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='spectra')
    gate, lock, jobs = threading.Lock(), threading.Lock(), {}
    app.extensions['spectra_executor'] = pool

    def job_dir(job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            return None
        directory = config.output_dir / job_id
        return directory if directory.is_dir() else None

    def run(job_id, sources):
        directory = config.output_dir / job_id
        try:
            with lock:
                jobs[job_id]['status'] = 'running'
            metadata = reconstruct(sources, directory, config, engine)
            with lock:
                jobs[job_id].update(status='complete', metadata=metadata)
        except Exception as exc:
            logging.exception('Reconstruction failed: %s', job_id)
            with lock:
                jobs[job_id].update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            with lock:
                (directory / 'job.json').write_text(json.dumps(jobs[job_id], indent=2))
            gate.release()

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get('/api/health')
    def health():
        import torch
        return jsonify(status='ok', cuda_available=torch.cuda.is_available(), config=config.json(),
                       pipegen={'verified': False, 'status': 'VGGT-only integration pending compatibility validation'})

    @app.post('/api/reconstruct')
    def submit():
        files = request.files.getlist('files')
        if not files or len(files) > config.max_frames:
            return jsonify(error=f'Upload one video or 1–{config.max_frames} images'), 400
        extensions = [Path(f.filename or '').suffix.lower() for f in files]
        if any(x not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS for x in extensions):
            return jsonify(error='Unsupported input file type'), 400
        if any(x in VIDEO_EXTENSIONS for x in extensions) and len(files) != 1:
            return jsonify(error='Upload a video alone'), 400
        if not gate.acquire(blocking=False):
            return jsonify(error='A reconstruction is already running; retry after it completes'), 409
        job_id = uuid4().hex
        directory = config.output_dir / job_id
        try:
            uploads = directory / 'uploads'
            uploads.mkdir(parents=True)
            sources = []
            for index, (file, extension) in enumerate(zip(files, extensions)):
                path = uploads / f'{index:04d}{extension}'
                file.save(path)
                sources.append(path)
            with lock:
                jobs[job_id] = {'id': job_id, 'status': 'queued'}
            pool.submit(run, job_id, sources)
        except Exception:
            gate.release()
            raise
        return jsonify(id=job_id, status='queued'), 202

    @app.get('/api/jobs/<job_id>')
    def status(job_id):
        directory = job_dir(job_id)
        if directory is None:
            return jsonify(error='Unknown job'), 404
        with lock:
            if job_id in jobs:
                return jsonify(jobs[job_id])
        if (directory / 'job.json').exists():
            return jsonify(json.loads((directory / 'job.json').read_text()))
        if (directory / 'metadata.json').exists():
            return jsonify(id=job_id, status='complete', metadata=json.loads((directory / 'metadata.json').read_text()))
        return jsonify(error='Job interrupted; resubmit inputs'), 409

    @app.get('/api/jobs/<job_id>/<filename>')
    def artifact(job_id, filename):
        if filename not in {'cloud.ply', 'metadata.json', 'predictions.npz', 'detections.jsonl'}:
            return jsonify(error='Unknown artifact'), 404
        directory = job_dir(job_id)
        if directory is None:
            return jsonify(error='Unknown job'), 404
        return send_from_directory(directory, filename, as_attachment=request.args.get('download') == '1')

    @app.get('/api/jobs/<job_id>/inspection')
    def inspection(job_id):
        directory = job_dir(job_id)
        if directory is None or not (directory / 'metadata.json').exists():
            return jsonify(error='A completed reconstruction is required'), 404
        path = directory / 'detections.jsonl'
        with lock:
            events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return jsonify(inspect_events(events))

    @app.route('/api/jobs/<job_id>/detections', methods=['GET', 'POST'])
    def detections(job_id):
        directory = job_dir(job_id)
        if directory is None or not (directory / 'metadata.json').exists():
            return jsonify(error='A completed reconstruction is required'), 404
        path = directory / 'detections.jsonl'
        if request.method == 'GET':
            with lock:
                events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            return jsonify(events=events, pipegen_execution_verified=False,
                           note='2D detections retained separately; no calibrated 3D association')
        try:
            event = validate_event(request.get_json())
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        with lock:
            with path.open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
        return jsonify(accepted=True, pipegen_execution_verified=False), 202

    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
