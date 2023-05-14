import os, uuid, base64, argparse, subprocess , queue
from io import BytesIO
from threading import Thread
from flask import Flask, request, jsonify
from PIL import Image
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
job_queue = queue.Queue()
job_status = {}
available_models = ['model1', 'model2', 'model3']

# Add command line arguments to enable Ngrok and apply a token
parser = argparse.ArgumentParser(description="Run Flask app with optional Ngrok and token.")
parser.add_argument("--ngrok", action="store_true", help="Enable Ngrok reverse tunneling")
parser.add_argument("--token", type=str, help="Use Ngrok auth token")

app_args = parser.parse_args()
  
if app_args.ngrok:
	if app_args.token:
		subprocess.check_call(["ngrok", "authtoken", app_args.token])
		run_with_ngrok(app)
	else:
		run_with_ngrok(app)

# Job queue
job_queue = queue.Queue()

# Job status dict
job_status = {}

# Available models list
available_models = ['model1', 'model2', 'model3']

# Helper function to save image from base64
def save_image_from_b64(b64_string, folder, filename):
	img_data = base64.b64decode(b64_string)
	img = Image.open(BytesIO(img_data))
	img_path = os.path.join(folder, filename)
	img.save(img_path)
	return img_path

def worker():
	while True:
		job = job_queue.get()
		if job is None:
			break
		job_id = job["job_id"]
		job_status[job_id] = "processing"
		process_job(job)
		job_status[job_id] = "completed"

# Start the worker thread
worker_thread = Thread(target=worker)
worker_thread.start()

@app.route('/api/status', methods=['GET'])
def status():
	# TODO: Return the status of jobs in queue, server idleness/workload, and GPU/RAM data
	pass

@app.route('/api/models', methods=['GET'])
def models():
	return jsonify(available_models)

@app.route('/api/job_status', methods=['GET'])
def get_job_status():
	job_id = request.args.get('jobid')
	if job_id and job_id in job_status:
		return jsonify(job_status[job_id])
	else:
		return jsonify({"error": "Job not found"}), 404

@app.route('/api/delete_job', methods=['DELETE'])
def delete_job():
	job_id = request.args.get('jobid')
	if job_id and job_id in job_status:
		job_status[job_id] = 'deleted'
		return jsonify({"result": "Job deleted"})
	else:
		return jsonify({"error": "Job not found"}), 404

@app.route('/api/<path:task>', methods=['POST'])
def create_task(task):
	if task in ['imagine', 'overpaint', 'inpaint', 'controlnet']:
		args = request.get_json()
		b64_string = args['source_image']
		temp_folder = 'temp'
		filename = f"{uuid.uuid4()}.png"
		img_path = save_image_from_b64(b64_string, temp_folder, filename)

		job_id = str(uuid.uuid4())
		job = {
			"job_id": job_id,
			"task": task,
			"img_path": img_path,
			"status": "queued",
			"args": args
		}

		job_queue.put(job)
		job_status[job_id] = job["status"]

		return jsonify({"job_id": job_id, "status": "queued"})
	else:
		return jsonify({"error": "Invalid task"}), 400

# Helper function to convert PIL image to base64
def image_to_base64(img):
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Task functions
def imagine(args):
	# TODO: Implement the 'imagine' function
	pass

def overpaint(args):
	# TODO: Implement the 'overpaint' function
	pass

def inpaint(args):
	# TODO: Implement the 'inpaint' function
	pass

def controlnet(args):
	# TODO: Implement the 'controlnet' function
	pass

# Function to process jobs
def process_job(job):
	task = job['task']
	args = job['args']
	result = None

	if task == 'imagine':
		result = imagine(args)
	elif task == 'overpaint':
		result = overpaint(args)
	elif task == 'inpaint':
		result = inpaint(args)
	elif task == 'controlnet':
		result = controlnet(args)

	if result is not None:
		b64_result = image_to_base64(result)
		job_status[job['job_id']] = {"status": "completed", "result": b64_result}
	else:
		job_status[job['job_id']] = {"status": "failed"}


if __name__ == '__main__':
	app.debug = False
	if app_args.ngrok:
		print('running with ngrok')
		run_with_ngrok(app)
		app.run()
	else:
		app.run(host='0.0.0.0', port=5000)