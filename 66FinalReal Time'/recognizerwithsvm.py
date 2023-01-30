# #############################################
# face detection with mtcnn on live cam feed  #
###############################################
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle, time


conf_t = 0.99

class recognizer:
    def __init__(self):
        self.facenet_model = load_model("facenet_keras.h5")
        self.svm_model = pickle.load(open("hasilsvmklasifikasi.sav", 'rb'))
        self.data = np.load('for5facesembeddings5november.npz')
        # object to the MTCNN detector class
        self.detector = MTCNN()
        self.ada_org = False

    def face_mtcnn_extractor(self, frame):
        """Methods takes in frames from video, extracts and returns faces from them"""
        # Use MTCNN to detect faces in each frame of the video
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        """Method takes the extracted faces and returns the coordinates"""
        # 1. Get the coordinates of the face
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        """Method takes in frame, face coordinates and returns preprocessed image"""
        # 1. extract the face pixels
        face = frame[y1:y2, x1:x2]
        # 2. resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        # 3. scale pixel values
        face_pixels = face_array.astype('float32')
        # 4. standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # 5. transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # 6. get face embedding
        yhat = self.facenet_model.predict(samples)
        face_embedded = yhat[0]
        # 7. normalize input vectors
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X
    
    def face_svm_classifier(self, X):
        """Methods takes in preprocessed images ,classifies and returns predicted Class label and probability"""
        # predict
        yhat = self.svm_model.predict(X)
        self.label = yhat[0]
        yhat_prob = self.svm_model.predict_proba(X)*100
        probability = round(yhat_prob[0][self.label], 3)
        trainy = self.data['arr_1']
        # predicted label decoder
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform(yhat)
        label = predicted_class_label[0]
        return label, str(probability)

    def sending(self):
        send = open("note", "w")
        name = self.label
        if self.ada_org:
            send.write("hallo ")
            send.write(str(name))
            print(" ngirim : ", name)
        else:
            send.write(" ")
        send.close()

    def face_detector(self):
        """Method classifies faces on live cam feed
            Class labels : sai_ram, donald_trump,narendra_modi, virat_koli"""
        # open cv for live cam feed
        cap = cv2.VideoCapture(0)
        new_frame_time = 0
        prev_frame_time = 0
        self.label = ""
        while True:
            # Capture frame-by-frame
            __, frame = cap.read()
            # 1. Extract faces from frames
            result = self.face_mtcnn_extractor(frame)
            self.ada_org = False        
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            fpstext = f'FPS : {int(fps)}'
            prob = 1
            cv2.putText(frame, fpstext, (7, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0), 2,
            lineType=cv2.LINE_AA)
            if result:
                for person in result:
                    self.ada_org = True
                    # 2. Localize the face in the frame
                    x1, y1, x2, y2, width, height = self.face_localizer(person)
                    # 3. Proprocess the images for prediction
                    X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(160, 160))
                    # 4. Predict class label and its probability
                    self.label, probability = self.face_svm_classifier(X)
                    confidence = person['confidence']
                    if confidence < conf_t:
                        continue
                    prob = confidence
                    # if(prob > 50):
                    # self.ada_org = True
                    print(" Person : {} , Probability : {}".format(self.label, probability))
                    key_points = person['keypoints'].values()
                    # 5. Draw a frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                    # 6. Add the detected class label to the frame
                    cv2.putText(frame, self.label.capitalize(), (x1, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 
                                lineType=cv2.LINE_AA)
                    cv2.putText(frame, probability, (x1, height + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 
                                lineType=cv2.LINE_AA)
                    for point in key_points:
                        cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)
                    # self.sending()

            # display the frame with label
            cv2.imshow('frame', frame)
            if (prob > 0.9):
                self.sending()
            # print(self.ada_org)
            # if(self.ada_org == False):
            #     self.sending()
            # break on keybord interuption with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything's done, release capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facedetector = recognizer()
    facedetector.face_detector()