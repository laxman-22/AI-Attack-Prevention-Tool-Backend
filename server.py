from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import base64
import os
import json
from PIL import Image
from io import BytesIO
from image_attack import fgsm, pgd, cw, deep_fool, preprocess_image
import torchvision
import torch
from model import predict

img_tensor = None
img_to_predict = None
model = torchvision.models.resnet34()
model.load_state_dict('resnet34-b627a593.pth')
model.eval()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://ai-attack-prevention-tool-website.vercel.app"}})
def load_labels():
    with open('imagenet-simple-labels.json', 'r') as f:
        return json.load(f)

@app.route('/getSampleImage', methods=['GET'])
def getSampleImage():
    try:
        isSampleSelected = request.args.get('sampleSelected', type=bool)

        if isSampleSelected:
            with open('goldfish.JPG', 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            res = jsonify({'image': encoded_image})
            response = make_response(res)
            return response, 200
        else:
            res = jsonify({"error": "Sample not selected"})
            response = make_response(res)
            return response, 400

    except Exception as e:
        res = jsonify({"error": str(e)})
        response = make_response(res)
        return response, 500 

@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    try:
        data = request.get_json()
        file = data.get('file')

        if not file:
            res = jsonify({"error": "No file provided"})
            response = make_response(res)
            return response, 400
        try:
            if file.startswith("data:image/"):
                file = file.split(",")[1]
            img_data = base64.b64decode(file)
            image = Image.open(BytesIO(img_data))
            image = image.convert('RGB')
            image_filename = os.path.join('uploads', "image_1.JPG")
            image.save(image_filename, 'JPEG')
            
            res = jsonify({"message": "File received and saved successfully", "filename": image_filename})
            response = make_response(res)
            return response, 200
                
        except Exception as e:
            res = jsonify({"error": f"Error decoding or saving the image: {str(e)}"})
            response = make_response(res)
            return response, 500
        
    except Exception as e:
        res = jsonify({"error": f"An error occurred: {str(e)}"})
        response = make_response(res)
        return response, 500
    
@app.route('/preprocessImage', methods=['POST'])
def preprocessImage():
    global img_tensor
    try:
        isSampleSelected = request.args.get('sampleSelected', 'false').lower() == 'true'
        if isSampleSelected:
            img = Image.open('goldfish.JPG').convert("RGB")
        else:
            img = Image.open('uploads/image_1.jpg').convert("RGB")

        img_tensor = preprocess_image(img)

        if img_tensor.shape[0] > 0:
            res = jsonify({"message": "Preprocessed Successfully"})
            response = make_response(res)
            return response, 200
        else:
            res = jsonify({"error": "Sample not selected"})
            response = make_response(res)
            return response, 400

    except Exception as e:
        print(f"Error: {str(e)}")
        res = jsonify({"error": f"Internal server error: {str(e)}"})
        response = make_response(res)
        return response, 500
    
@app.route('/attackImage', methods=['POST'])
def attackImage():
    global img_tensor
    global img_to_predict

    try:
        data = request.get_json()
        if img_tensor.shape[0] > 0:
            if data.get('attackType') == 'No Attack':
                img_to_predict = img_tensor
                res = jsonify({"message": "No Attack Performed"})
                response = make_response(res)
                return response, 200
            elif data.get('attackType') == "FGSM":
                labels = list(map(str, load_labels()))
                label_input = str(data.get('label'))
                index = labels.index(label_input)
                epsilon = float(data.get('epsilon', 0.02))
                attacked = fgsm(model=model, images=img_tensor, label=torch.tensor([index]), epsilon=epsilon)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float()
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)
                buffered = BytesIO()
                attacked_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                res = jsonify({"attacked_image_base64": img_base64})
                response = make_response(res)
                
                return response, 200
            elif data.get('attackType') == 'PGD':
                labels = list(map(str, load_labels()))
                label_input = str(data.get('label'))
                index = labels.index(label_input)
                epsilon = float(data.get('epsilon', 0.02))
                alpha = float(data.get('alpha', 0.01))
                iterations = int(data.get('iterations', 50))
                attacked = pgd(model=model, images=img_tensor, label=torch.tensor([index]), epsilon=epsilon, alpha=alpha, iterations=iterations)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float()
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)
                buffered = BytesIO()
                attacked_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                res = jsonify({"attacked_image_base64": img_base64})
                response = make_response(res)
                return response, 200
            elif data.get('attackType') == "C&W":
                labels = list(map(str, load_labels()))
                label_input = str(data.get('label'))
                index = labels.index(label_input)
                confidence = float(data.get('confidence', 1))
                learningRate = float(data.get('learningRate', 0.01))
                iterations = int(data.get('iterations', 50))
                attacked = cw(model=model, images=img_tensor, label=torch.tensor([index]), confidence=confidence, learning_rate=learningRate, iterations=iterations)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float()
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)
                buffered = BytesIO()
                attacked_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                res = jsonify({"attacked_image_base64": img_base64})
                response = make_response(res)
                return response, 200
            elif data.get('attackType') == 'DeepFool':
                labels = list(map(str, load_labels()))
                label_input = str(data.get('label'))
                index = labels.index(label_input)
                overshoot = float(data.get('overshoot', 0.02))
                iterations = int(data.get('iterations', 50))
                attacked = deep_fool(model=model, images=img_tensor, label=torch.tensor([index]), overshoot=overshoot, iterations=iterations)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float()
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)
                buffered = BytesIO()
                attacked_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                res = jsonify({"attacked_image_base64": img_base64})
                response = make_response(res)
                return response, 200
            else: 
                res = jsonify({"error": "Unknown Attack Type"})
                response = make_response(res)
                return response, 400
        else:
            res = jsonify({"error": "Sample not selected"})
            response = make_response(res)
            return response, 400

    except Exception as e:
        res = jsonify({"error": str(e)})
        response = make_response(res)
        return response, 500

@app.route('/generatePrediction', methods=['GET'])
def generatePrediction():
    try:
        binary_pred, attack_pred = predict(img_to_predict)
        attack_mapping = {
            0: "no_attack",
            1: "deepfool",
            2: "fgsm",
            3: "pgd",
            4: "cw"
        }
        attack_type = attack_mapping.get(attack_pred, "unknown")
        print(f"Binary prediction (0 = clean, 1 = attacked): {binary_pred}")
        print(f"Attack prediction (0 = no_attack, 1 = deepfool, 2 = fgsm, 3 = pgd, 4 = cw): {attack_type}")
        
        res = jsonify({"isClean": binary_pred, "attackType": attack_type})
        response = make_response(res)
        return response, 200
    except Exception as e:
        res = jsonify({"error": str(e)})
        response = make_response(res)
        return response, 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)