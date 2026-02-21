# 🌿 Plant Classification System

A web-based plant identification tool powered by deep learning. Upload an image of a plant, and the system identifies the species with confidence scores plus detailed botanical information.

## What I Built

I wanted to learn how CNNs actually work in production, so I trained a model to recognize 15 different plant species and deployed it as a real web app. The interesting part was figuring out how to go from a trained model to something anyone could use through their browser.

**What it does:**
- Identifies 15 plant species from uploaded images
- Provides confidence scores for predictions
- Shows botanical information (scientific name, genus, habitat)
- Works through a simple web interface
- Processes images in real-time (~6-7 seconds)

**Real stats:** The CNN achieves around 85-87% accuracy on the test set, which is pretty solid for a relatively small model running on CPU.

---

## How It Works

**The Simple Version:**
1. You upload a plant image through the web interface
2. The Flask backend receives and preprocesses it (resize to 256x256, normalize)
3. The trained CNN model (stored as .h5 file) analyzes the image
4. Model outputs probabilities for all 15 species
5. System picks the highest probability and looks up botanical info from Excel
6. Results display with species name, confidence, and plant details

**Why This Architecture?**
- **CNN model** = The brain that learned plant features during training
- **Flask backend** = Handles image uploads and serves predictions
- **Gevent WSGI** = Production-ready server (not Flask's dev server)
- **Excel database** = Simple storage for botanical information

**Key Design Choice:** I separated the model training from deployment. The .h5 file contains all the learned weights, so the web app just loads it once on startup and reuses it for all predictions.

---

## Plant Species Supported

The model recognizes these 15 species:
- Jade Plant
- Lucky Bamboo
- Venus Fly Trap
- Zebra Plant
- Poinsettia
- String of Bananas
- Paddle Plant
- Nerve Plant
- Moon Cactus
- House Leek
- Elephant Ear
- Coleus
- Begonia Maculata
- Acer Capillipes
- Acer Circinatum

---

## Tech Stack

**Machine Learning:**
- TensorFlow/Keras for the CNN model
- Python 3.11
- NumPy for array operations

**Web Application:**
- Flask as the web framework
- Gevent WSGI server for production
- HTML/CSS/JavaScript for the frontend
- Base64 encoding for image transfer

**Data Management:**
- Pandas for reading Excel database
- Excel file with botanical information

---

## Getting Started

**Requirements:**
```bash
pip install tensorflow flask gevent pandas pillow numpy
```

**Run the app:**
```bash
python app.py
```

Then open your browser to `http://localhost:5000`

---

## Project Structure

```
plant-classification-system/
├── app.py                          # Flask application
├── UrbanPlantClassifier.h5         # Trained CNN model
├── plantsDescription.xlsx          # Botanical information database
├── templates/
│   └── index.html                  # Web interface
└── static/
    └── css/
        └── style.css               # Styling
```

---

## How the CNN Works

**Model Architecture:**
The model is a convolutional neural network trained to recognize visual patterns in plant images. Here's what happens during prediction:

1. **Input Layer:** Receives 256x256 RGB image (3 color channels)
2. **Convolutional Layers:** Extract features like edges, textures, leaf patterns
3. **Pooling Layers:** Reduce dimensions while keeping important features
4. **Dense Layers:** Combine features to make final classification
5. **Output Layer:** 15 neurons with softmax activation (one per species)

**Training Process:**
- Images were resized and normalized (pixel values 0-1)
- Used ReLU activation for non-linearity
- Applied dropout to prevent overfitting
- Trained with categorical crossentropy loss
- Optimized with Adam optimizer

---

## Performance

Real numbers from my testing:
- **Accuracy:** 85-87% on validation set
- **Inference time:** 6-7 seconds per image (CPU)
- **Model size:** Compact enough to run on free hosting
- **Input format:** Any common image format (JPEG, PNG, etc.)
- **Image size:** Automatically resized to 256x256

**What affects accuracy:**
- Image quality and lighting
- How clearly the plant is visible
- Whether the plant looks similar to training images
- Background clutter in the photo

---

## What Makes This Interesting

**Real Machine Learning Deployment:**
This isn't just a Jupyter notebook - it's a complete ML pipeline from training to production. The .h5 file is the actual trained model with all learned weights. Without it, the app can't classify anything.

**End-to-End Pipeline:**
I handled everything: data preprocessing, model training, evaluation, deployment, and building the web interface. This gave me hands-on experience with the full ML lifecycle.

**Production Considerations:**
Used Gevent WSGI server instead of Flask's development server, implemented proper error handling, and made sure the model loads efficiently on startup.

---

## What I Learned

Building this taught me:
- **CNNs in practice** - How convolutional layers actually extract features
- **Model deployment** - Loading .h5 files and serving predictions via API
- **Image preprocessing** - Critical importance of matching training specifications
- **Flask integration** - Connecting ML models with web frameworks
- **Trade-offs** - Balancing model complexity vs inference speed

**Challenges I solved:**
- Handling different image formats and sizes from users
- Ensuring preprocessing matches training pipeline exactly
- Managing TensorFlow/Keras version compatibility
- Building a clean interface for non-technical users

---

## How to Test It

**Try these plants:**
1. Upload a clear image of the plant
2. Make sure the plant is the main focus
3. Avoid extreme angles or lighting
4. Wait ~6-7 seconds for processing

**The app returns:**
- Predicted species name
- Confidence score (0-100%)
- Scientific name
- Genus classification
- Natural habitat information

---

## Possible Improvements

If I were to extend this project:
- Add more plant species (currently limited to 15)
- Implement transfer learning (ResNet, EfficientNet) for better accuracy
- Add data augmentation during training (rotations, flips, brightness changes)
- Create a mobile app version for field use
- Implement caching for faster repeated predictions
- Add user feedback to improve the model over time
- Support for identifying multiple plants in one image

---

## Why This Project Matters

**For Learning:**
This project bridges the gap between learning CNNs in theory and deploying them in practice. Understanding how to go from `model.fit()` to a production web app is crucial for ML engineering roles.

**For Users:**
Plant identification can help gardeners, botanists, and plant enthusiasts quickly learn about species they encounter. The added botanical information makes it educational.

**For My Portfolio:**
Demonstrates ability to train deep learning models, deploy them in production, and build complete applications - not just run tutorial notebooks.

---

## Technical Details for Interviews

**Model Training:**
- Used categorical crossentropy loss (multi-class classification)
- Adam optimizer with learning rate scheduling
- Validation split to monitor overfitting
- Early stopping to prevent overtraining

**Deployment:**
- Model loaded once on server startup (efficient)
- Images preprocessed to match training specifications
- Predictions served via REST API
- Gevent WSGI server for concurrent requests

**Data Pipeline:**
- Images resized to 256x256 pixels
- Pixel values normalized to [0, 1]
- RGB format maintained
- Excel lookup for botanical information

**Built to learn CNNs in production. If this helps you learn too, give it a star! ⭐**
