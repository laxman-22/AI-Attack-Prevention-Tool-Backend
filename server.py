from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import base64
import os
import json
from PIL import Image
from io import BytesIO
from image_attack import fgsm, pgd, cw, deep_fool, preprocess_image
import torch
import torchvision.models as models
from model import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize global variables
img_tensor = None
img_to_predict = None
model = models.resnet34(weights='IMAGENET1K_V1')  
model = model.to(device)
model.eval()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def load_labels():
    '''
    This function loads the imagenet labels in order to perform attacks
    
    Returns:
    json.load(f) (JSON): the json result of laoding the file
    '''
    with open('imagenet-simple-labels.json', 'r') as f:
        return json.load(f)

@app.route('/getSampleImage', methods=['GET'])
@cross_origin(origins="https://ai-attack-prevention-tool-website.vercel.app")
def getSampleImage():
    '''
    This function returns the sample goldfish image in case users don't want to upload their own image

    Returns:
    response (JSON): returns a JSON response to the front end
    '''
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
@cross_origin(origins="https://ai-attack-prevention-tool-website.vercel.app")
def uploadImage():
    '''
    This function uploads the user submitted image to the server so actions can be performed on it

    Returns:
    response (JSON): returns a JSON response to the front end
    '''
    try:
        data = request.get_json()
        file = data.get('file')

        if not file:
            res = jsonify({"error": "No file provided"})
            response = make_response(res)
            return response, 400
        try:
            # Decode the base64 encoded image to save it
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
@cross_origin(origins="https://ai-attack-prevention-tool-website.vercel.app")
def preprocessImage():
    '''
    This function performs any pre processing required before attacking the image

    Returns:
    response (JSON): returns a JSON response to the front end
    '''
    global img_tensor
    try:
        isSampleSelected = request.args.get('sampleSelected', 'false').lower() == 'true'
        # Select image based on whether or not the user selected the sample
        if isSampleSelected:
            img = Image.open('goldfish.JPG').convert("RGB")
        else:
            img = Image.open('uploads/image_1.jpg').convert("RGB")

        # Preprocess the image
        img_tensor = preprocess_image(img)

        # Return an appropriate response
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
@cross_origin(origins="https://ai-attack-prevention-tool-website.vercel.app")
def attackImage():
    '''
    This function performs the attacks based on attack type provided by the front-end
    Returns:
    response (JSON): returns a JSON response to the front-end
    '''
    global img_tensor
    global img_to_predict

    try:
        data = request.get_json()
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224))
        if img_tensor.shape[0] > 0:
            if data.get('attackType') == 'No Attack':
                img_to_predict = img_tensor.to(device)
                res = jsonify({"message": "No Attack Performed"})
                response = make_response(res)
                return response, 200
            elif data.get('attackType') == "FGSM":
                # Ensure correct types
                labels = list(map(str, load_labels()))
                label_input = str(data.get('label'))
                index = labels.index(label_input)
                epsilon = float(data.get('epsilon', 0.02))
                img_tensor = img_tensor.to(device)
                # Perform the attack
                with torch.no_grad():
                    attacked = fgsm(model=model, images=img_tensor, label=torch.tensor([index]), epsilon=epsilon)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                # Save the image to make predictions on
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                # Prepare to send the attacked image back
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)
                buffered = BytesIO()
                attacked_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Return a response
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
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    attacked = pgd(model=model, images=img_tensor, label=torch.tensor([index]), epsilon=epsilon, alpha=alpha, iterations=iterations)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
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
                img_tensor = img_tensor.to(device)
                
                with torch.no_grad():
                    attacked = cw(model=model, images=img_tensor, label=torch.tensor([index]), confidence=confidence, learning_rate=learningRate, iterations=iterations)
                
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
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
                img_tensor = img_tensor.to(device)

                attacked = deep_fool(model=model, images=img_tensor, label=torch.tensor([index]), overshoot=overshoot, iterations=iterations)
                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                
                img_to_predict = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
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
@cross_origin(origins="https://ai-attack-prevention-tool-website.vercel.app")
def generatePrediction():
    '''
    This function generates the predictions for the attacked image

    Returns:
    response (JSON): returns a JSON response to the front end
    '''
    try:
        if img_to_predict is None:
            res = jsonify({"error": "No image has been provided or set for prediction"})
            response = make_response(res)
            return response, 400
        # Create a user friendly mapping
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
        
        # Return response
        res = jsonify({"isClean": binary_pred, "attackType": attack_type})
        response = make_response(res)
        return response, 200
    except Exception as e:
        res = jsonify({"error": str(e)})
        response = make_response(res)
        return response, 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)