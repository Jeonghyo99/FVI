from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os
import glob

from train import eval_only
import torch


UPLOAD_FOLDER = '/content/FVI/data/real'  # 코랩 파일 시스템에 업로드 폴더를 설정
ALLOWED_EXTENSIONS = {'wav'}  # 허용되는 파일 확장자

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run_with_ngrok(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully'


@app.route('/evaluate', methods=['GET'])
def evaluate():
    fakecounter = eval_only(
        real_dir = '/content/FVI/data/real',
        fake_dir = '/content/FVI/data/fake',
        device = "cpu",
        batch_size = 128,
        feature_classname = "mfcc",
        model_classname = "ShallowCNN",
        in_distribution = True,
        checkpoint=torch.load('/content/FVI/saved/ShallowCNN_mfcc_I/best.pt', map_location=torch.device('cpu')),
    )

    return jsonify({"fakecounter": str(fakecounter)})


@app.route('/clean', methods=['GET'])
def clean():
    # '/content/FVI/data/real' 폴더에 있는 모든 wav 파일을 찾습니다.
    file_list = glob.glob('/content/FVI/data/real/*.wav')

    for file_path in file_list:
        # 'segment_11111.wav' 파일을 제외하고 나머지 파일을 삭제합니다.
        if os.path.basename(file_path) != 'output_0002.wav':
            os.remove(file_path)

    return 'Clean up complete'


@app.route("/")
def home():
    return "Hello, world!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")