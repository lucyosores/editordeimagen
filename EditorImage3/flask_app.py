from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os

app = Flask(__name__)

# Asegúrate de crear un directorio temporal para guardar imágenes
if not os.path.exists('static'):
    os.makedirs('static')

# Transformaciones de la imagen
transformaciones = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

modelo = models.segmentation.deeplabv3_resnet101(pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')
modelo.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'imagen' not in request.files:
        return 'No image uploaded', 400

    file = request.files['imagen']
    imagen = Image.open(file.stream)

    # Procesar la imagen
    imagen_tensor = transformaciones(imagen).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Aplicar el modelo a la imagen
    with torch.no_grad():
        output = modelo(imagen_tensor)['out'][0]

    # Escalar los valores de salida a [0,1]
    output = output.detach().cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min())
    
    # Obtener la clase con la mayor probabilidad
    predicted_class = np.argmax(output, axis=0)

    # Crear un mapa de colores
    color_map = plt.cm.get_cmap('viridis', 21)
    output_rgb = color_map(predicted_class)[:, :, :3]

    # Convertir a imagen PIL
    output_imagen = Image.fromarray((output_rgb * 255).astype('uint8'))
    output_imagen.save('static/imagen_editada.png')

    return render_template('index.html', resultado=True)

@app.route('/static/<path:filename>')
def send_image(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
