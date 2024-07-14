import os
import time
import glob
import subprocess
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
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


@app.route('/file_upload')
def home():
    return jsonify("Hello, This is a file-upload API, To send the file, use http://13.213.81.139/file_upload/send_file")


@app.route('/file_upload/status1', methods=['POST'])
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
        print(filename)
        print(file)
        app.logger.debug("Spreadsheet received successfully")
        response = jsonify({'message': 'Spreadsheet uploaded successfully'})
        response.status_code = 201
        return response
    else:
        app.logger.debug("Allowed file types are csv or xlsx")
        response = jsonify({'message': 'Allowed file types are csv or xlsx'})
        response.status_code = 400
        return response


@app.route('/file_upload/status2', methods=['POST'])
def status1():
    global filenames
    app.logger.debug("file_upload/status2 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            filenames = request.get_json()
            app.logger.debug(filenames)
            filenames = filenames['data']
            # print(filenames)
            folderpath = glob.glob('C:\\inetpub\\wwwroot\\iAssist_IT_support\\New_IT_support_datasets\\*.csv')
            latest_file = max(folderpath, key=os.path.getctime)
            # print(latest_file)
            time.sleep(3)
            if filenames in latest_file:
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
    return jsonify('New data merged with existing datasets')


@app.route('/file_upload/status3', methods=['POST'])
def status2():
    app.logger.debug("file_upload/status3 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
    return jsonify("New model training is in progress don't upload new file")


@app.route('/file_upload/status4', methods=['POST'])
def model_run():
    app.logger.debug("file_upload/status4 route is executed")
    if request.method == 'POST':
        # Get data in json format
        if request.get_json():
            message = request.get_json()
            message = message['data']
            app.logger.debug(message)
            app.logger.debug(currentDateTime)
            with open("model-creation-status.log", 'a') as f:
                app.logger.debug(subprocess.run("python C:\\Users\\kavin\\Documents\\IT_support_chatbot-master\\Python_files\\main.py", shell=True, stdout=f, text=True))
    return jsonify("Model created successfully for sent file %s" % filenames)


if __name__ == "__main__":
    app.run()
