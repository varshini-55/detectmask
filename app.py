import uuid
from flask import Flask, render_template, Response, redirect,flash, request # type: ignore
from imutils.video import VideoStream # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type:ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import imutils # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os
import time

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
vs = None
streaming = False

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

def start_video_stream():
    global vs, streaming
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    streaming = True
    while streaming:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cv2.destroyAllWindows()
    vs.stop()

def detect_and_predict_image(image_path):
    # Load face detector and mask detector models
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")

    # Load the input image from disk
    image = cv2.imread(image_path)

    # If the image is None, return None (indicating failure)
    if image is None:
        return None

    # Get the spatial dimensions of the image
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the image
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine if the face has a mask or not
            (mask, withoutMask) = maskNet.predict(face)[0]

            # Determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)   
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Save the processed image with a unique filename
    processed_image_filename = os.path.join(app.config['STATIC_FOLDER'], 'processed_image_' + uuid.uuid4().hex + '.jpg')
    cv2.imwrite(processed_image_filename, image)
    
    # Return the path to the processed image
    return processed_image_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        # Save the uploaded image to a static folder
        image_path = os.path.join(app.config['STATIC_FOLDER'], file.filename)
        file.save(image_path)

        # Perform face mask detection on the uploaded image
        processed_image_path = detect_and_predict_image(image_path)
        print(processed_image_path)
        return render_template('result.html', image_path=processed_image_path)


@app.route('/video_feed')
def video_feed():
    return Response(start_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global vs,streaming
    streaming = False
    if vs:
        vs.stop()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
