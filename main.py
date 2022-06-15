import pathlib
from hmac import compare_digest
from model.user import User
from flask import Flask
from flask_jwt import *
from werkzeug.utils import secure_filename
import hashlib
import os
import torch
import clip
from PIL import Image
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
classes = ["grapefruit", "lemon", "lime", "orange", "tangerine"]
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

mainFolder = pathlib.Path(__file__).parent.absolute()
uploadsFolder = os.path.join(mainFolder, "uploads")
securityFolder = os.path.join(mainFolder, "security")
allowdExtensions = {"jpg", "jpeg"}

username_table = {}
userid_table = {}

try:
    users = []
    credentials = credentials.Certificate(os.path.join(securityFolder, "firebase_admin.json"))
    firebase_admin.initialize_app(credentials)
    page = auth.list_users()
    index = 0
    while page:
        for user in page.users:
            users.append(
                User(
                    index, 
                    hashlib.sha3_256(user.email.encode("utf-8")).hexdigest(), 
                    hashlib.sha3_256(user.uid.encode("utf-8")).hexdigest()
                )
            )
            index += 1
        page = page.get_next_page()
    username_table = {u.username: u for u in users}
    userid_table  = {u.id: u for u in users}
except Exception as e:
    print(e)

app = Flask(__name__)
app.config["SECRET_KEY"] = "" #RANDOM

def auth(username:str, password:str):
    uHash = hashlib.sha3_256(username.encode("utf-8")).hexdigest()
    pHash = hashlib.sha3_256(password.encode("utf-8")).hexdigest()
    user = username_table.get(uHash, None)
    if user and compare_digest(user.password.encode("utf-8"), pHash.encode("utf-8")):
        return user

def identity(payload):
    userId = payload['identity']
    return userid_table.get(userId, None)

if not os.path.exists(uploadsFolder):
    os.makedirs(uploadsFolder)

def allowed_file(filename:str):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowdExtensions

def remove(file_path:str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        print ("Error: {} - {}".format(e.filename, e.strerror))

def inference(file_path:str):
    try:
        image_input = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(1)
        return {"response" : "{}".format(classes[indices[0].item()])}, 200
    except:
        return {"response" : "Internal Server Error"}, 500
    finally:
        remove(file_path)

jwt = JWT(app, authentication_handler=auth, identity_handler=identity)

@app.route("/clip", methods=["POST"])
@jwt_required()
def clipRoute():
    if request.method == "POST":
        if "file" not in request.files:
            return {"response" : "No file part"}, 400

        file = request.files["file"]

        if file.filename == "":
            return {"response" : "No file selected"}, 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "{}_{}.{}".format(current_identity,datetime.today().strftime('%Y%m%d%H%M%S'),filename.rsplit('.', 1)[1].lower())
            filename = secure_filename(filename)
            file_path = os.path.join(uploadsFolder, filename)
            if not os.path.exists(file_path):
                file.save(file_path)
                return inference(file_path)
            else:
                return {"response" : "File already exists"}, 500
        else:
            return {"response" : "File format not allowed"}, 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000, debug=True, ssl_context=(os.path.join(securityFolder, "cert.pem"), os.path.join(securityFolder, "key.pem")))