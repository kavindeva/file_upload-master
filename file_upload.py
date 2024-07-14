import io
import os
import time
import glob
import base64
import shutil
import asyncio
import logging
import subprocess
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

# from flask_cors import cross_origin

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
_executor = ThreadPoolExecutor(1)

app = Flask(__name__)
# file_uploadCorsConfig = {"origins": "117.203.77.148"}
# CORS(app, resources={r"/file_upload/*": file_uploadCorsConfig})
app.config['UPLOAD_FOLDER'] = "C:\\inetpub\\wwwroot\\iAssist_IT_support\\New_IT_support_datasets"
currentDateTime = datetime.now()
filenames = None

logger = logging.getLogger(__name__)
app.logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('model-creation-status.log')
file_handler.setFormatter(formatter)

# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)

app.logger.addHandler(file_handler)


# app.logger.addHandler(stream_handler)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/create', methods=['GET'])
# @cross_origin(allow_headers=['Content-Type'])
def home():
    app.logger.debug("file_upload route")
    data = "Welcome to Model creation Application Programming Interface"
    return jsonify(data)


@app.route('/api/create/status1', methods=['POST'])
def upload_file():
    app.logger.debug("/file_upload/status1 is execution")
    # check if the post request has the file part
    if 'file' not in request.files:
        app.logger.debug("No file part in the request")
        response = jsonify({'message': 'No file part in the request'})
        response.status_code = 400
        return response
    file = request.files['file']
    if file.filename == '':
        app.logger.debug("No file selected for uploading")
        response = jsonify({'message': 'No file selected for uploading'})
        response.status_code = 400
        return response
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print(filename)
        # print(file)
        app.logger.debug("Spreadsheet received successfully")
        response = jsonify({'message': 'Spreadsheet uploaded successfully'})
        response.status_code = 201
        return response
    else:
        app.logger.debug("Allowed file types are csv or xlsx")
        response = jsonify({'message': 'Allowed file types are csv or xlsx'})
        response.status_code = 400
        return response


@app.route('/api/create/status2', methods=['POST'])
def status1():
    global filenames
    app.logger.debug("file_upload/status2 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            userData = request.get_json()
            app.logger.debug(filenames)
            filenames = userData['filename']
            # print(filenames)
            userName = userData['email']
            # print(filenames)
            folderpath = glob.glob('C:\\inetpub\\wwwroot\\iAssist_IT_support\\New_IT_support_datasets\\*.csv')
            latest_file = max(folderpath, key=os.path.getctime)
            # print(latest_file)
            time.sleep(3)
            if filenames and latest_file:
                df1 = pd.read_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\New_IT_support_datasets\\" +
                                  filenames, names=["errors", "solutions"])
                df1 = df1.drop(0)
                # print(df1.head())
                df2 = pd.read_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\existing_tickets.csv",
                                  names=["errors", "solutions"])
                combined_csv = pd.concat([df2, df1])
                combined_csv.to_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\new_tickets-chatdataset.csv",
                                    index=False, encoding='utf-8-sig')
                time.sleep(2)
                # return redirect('/file_upload/status2')
            datasetDirectory = Path("C:\\inetpub\\wwwroot\\iAssist_IT_support\\New_IT_support_datasets")
            userDirectory = datasetDirectory / userName
            # print(userDirectory)
            if os.path.isdir(userDirectory):
                # print("True")
                app.logger.debug(latest_file)
                app.logger.debug(userDirectory)
                shutil.move(latest_file, userDirectory / filenames)
            else:
                # print("False")
                os.mkdir(os.path.join(datasetDirectory, userName))
                app.logger.debug(latest_file)
                app.logger.debug(userDirectory)
                shutil.move(latest_file, userDirectory)
    return jsonify('New data merged with existing datasets')


@app.route('/api/create/status3', methods=['POST'])
def status2():
    app.logger.debug("file_upload/status3 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
    return jsonify("New model training is in progress don't upload new file")


@app.route('/api/create/status4', methods=['POST'])
def model_run():
    app.logger.debug("file_upload/status4 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
            app.logger.debug(currentDateTime)

            async def model_creation():
                app.logger.debug("model script executed to run")
                app.logger.debug(subprocess.run("python C:\\inetpub\\wwwroot\\file_upload\\main.py", shell=True))
                app.logger.debug("script ran successfully")

            asyncio.run(model_creation())
    return jsonify("Model created successfully for sent file.")


@app.route('/api/create/accuracy', methods=['POST'])
def model_result1():
    app.logger.debug("/api/create/accuracy route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
            folderpath1 = glob.glob('C:\\inetpub\\wwwroot\\iAssist_IT_support\\Model_results\\*.png')
            latest_file = sorted(folderpath1, key=os.path.getmtime)
            # imageName = latest_file[0]
            accuracyImage = Image.open(latest_file[0], mode='r')
            img_byte_arr = io.BytesIO()
            accuracyImage.save(img_byte_arr, format='PNG')
            my_encoded_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
            response_data = {"image": my_encoded_image}
            return jsonify(response_data)
            # bytesImage = accuracyImage.tobytes()
            # response = make_response(bytesImage)
            # response.headers.set('Content-Type', 'image/png')
            # response.headers.set(
            #     'Content-Disposition', 'attachment', filename='%s.png' % bytesImage)
            # response.headers['Content-Transfer-Encoding'] = 'base64'
            # return response
            # return send_file(io.BytesIO(bytesImage), as_attachment=True, attachment_filename='%s.png' % accuracyImage,
            # mimetype='image/png')


@app.route('/api/create/loss', methods=['POST'])
def model_result2():
    app.logger.debug("/api/create/loss route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
            folderpath1 = glob.glob('C:\\inetpub\\wwwroot\\iAssist_IT_support\\Model_results\\*.png')
            latest_file = sorted(folderpath1, key=os.path.getmtime)
            # imageName = latest_file[1]
            accuracyImage = Image.open(latest_file[1], mode='r')
            img_byte_arr = io.BytesIO()
            accuracyImage.save(img_byte_arr, format='PNG')
            my_encoded_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
            response_data = {"image": my_encoded_image}
            return jsonify(response_data)
            # return send_file(imageName, as_attachment=True, download_name=imageName, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True, port=8088)
