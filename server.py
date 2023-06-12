import os, uuid, base64, argparse, subprocess , queue
from io import BytesIO
from threading import Thread
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
from PIL import Image
from IPython import display as disp

from sd import SD, download_models


job_queue = queue.Queue()
job_status = {}

import subprocess

parser = argparse.ArgumentParser(description="Run Flask app with Ngrok")
parser.add_argument("--token", type=str, help="Use Ngrok auth token")
parser.add_argument("--models_path", type=str, default='/content/models/', help="Path to models directory")
parser.add_argument("--log", action="store_true", help="log mode")

app_args = parser.parse_args()


# init
app = Flask(__name__)
# Setting the authtoken (optional)
if app_args.token:
	conf.get_default().auth_token =  app_args.token
##
public_url = ngrok.connect(5000)
print("Public URL:", public_url)
#
models_path = app_args.models_path # set model path variable
temp_folder = 'temp'
if not os.path.exists(models_path):
	os.makedirs(models_path)
if not os.path.exists(temp_folder):
	os.makedirs(temp_folder)

available_models = [
	'CompVis/stable-diffusion-v1-4',
	#'runwayml/stable-diffusion-v1-5',
	#'dreamlike-art/dreamlike-photoreal-2.0',
	#''
	]

############# xxx ##############
def clear():
    disp.clear_output()
def init(models):
	download_models(models,models_path)
	sd = SD( models_path, models[0], None)
	return sd
	
sd = init(available_models)

# Job queue
job_queue = queue.Queue()

# Job status dict
job_status = {}

# Available models list
available_models = ['model1', 'model2', 'model3']


def save_image_from_b64(b64_string, folder, filename):
    print(b64_string)
    img_data = base64.urlsafe_b64decode(b64_string)
    img_file_path = os.path.join(folder, filename)
    
    # Save the raw data to a file for debugging
    with open(img_file_path + ".raw", "wb") as raw_file:
        raw_file.write(img_data)
    
    try:
        img = Image.open(BytesIO(img_data))
        img_path = os.path.join(folder, filename)
        img.save(img_path)
        return img_path
    except Exception as e:
        print("Exception when trying to open image: ", e)
        return None

	
def log(message):
	if app_args.log:
		print('####################################')
		print(message)
		print('####################################')
	
def worker():
	while True:
		job = job_queue.get()
		if job is None:
			break
		process_job(job)
		
# Start the worker thread
worker_thread = Thread(target=worker)
worker_thread.start()

@app.route('/api/info/status', methods=['GET'])
def status():
	# TODO: Return the status of jobs in queue, server idleness/workload, and GPU/RAM data
	pass

@app.route('/api/info/models', methods=['GET'])
def models():
	return jsonify(available_models)

@app.route('/api/job/delete', methods=['DELETE'])
def delete_job():
	job_id = request.args.get('jobid')
	if job_id and job_id in job_status:
		job_status[job_id]['status'] = 'deleted'
		return jsonify({"result": "Job deleted"})
	else:
		return jsonify({"error": "Job not found"}), 404


@app.route('/api/job/<path:job_id>/status', methods=['GET'])
def get_job_status(job_id):
	log(job_id)
	if job_id and job_id in job_status:
		return jsonify(job_status[job_id])
	else:
		return jsonify({"error": "Job not found"}), 404


@app.route('/api/<path:task>', methods=['POST'])
def create_task(task):
	log(task) ############## x ###############
	if task in ['imagine', 'overpaint', 'inpaint', 'controlnet']:
		jobjson = request.get_json()
		args = jobjson['args']
		job_id = jobjson['id']
		
		log(args) ############## x ###############
		
		if (task!='imagine'):
			b64_string = args['initImage']
			filename = f"{job_id}.png"
			#img_path = save_image_from_b64(b64_string, temp_folder, filename)
			f = BytesIO()
			f.write(base64.b64decode(b64_string))
			f.seek(0)
			
			if inpaint=="true":
				img = Image.open(f)
			else:
				img = Image.open(f).convert("RGB")
			img.save(filename)
			img_path = filename
		else:
			img_path = None

		#job_id = str(uuid.uuid4())
		args['img_path'] =  img_path
		job = {
			"job_id": job_id,
			"task": task,
			"status": "queued",
			"args": args
		}

		job_queue.put(job)
		
		job_status[job_id] = {"job_id": job_id, "status": "queued"}

		return jsonify(job_status[job_id])
	else:
		return jsonify({"error": "Invalid task"}), 400

# Helper function to convert PIL image to base64
def image_to_base64(img):
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")

# SD functions
def imagine(args):
	return sd.txt2img(args['prompt'],num_inference_steps=args['steps'])[0][0]

def overpaint(args):
	return sd.img2img(args['prompt'],Image.open(args['img_path']).convert('RGB'),num_inference_steps=args['steps'])[0][0]
	
def inpaint(args):
	return sd.txt2img(args['prompt'],args['img_path'],num_inference_steps=args['steps'])[0][0]
	
def controlnet(args):
	return sd.txt2img(args['prompt'],args['img_path'],num_inference_steps=args['steps'])[0][0]
	
# Function to process jobs
def process_job(job):
	task = job['task']
	args = job['args']
	job_id = job["job_id"]
	job_status[job_id]['status'] = "processing"
	
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


if __name__ == "__main__":
    clear()
    log(public_url)
    app.run()
