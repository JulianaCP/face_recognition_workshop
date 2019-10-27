import os
import pickle
import numpy as np
import cv2
from PIL import Image
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN

class BaseFaceExtractor:
    '''
        Class to extract and encode faces 
    '''
    def __init__(self, facenet_model, cascade_classifier):
        self.facenet = facenet_model
        self.face_detector = MTCNN()
        self.cascade_classifier = cascade_classifier

    def extract_faces_opencv(self, img, min_size=(100, 100)):
        ''' 
            Extract all the faces in the images.
            It uses the cascadeClassifier to find the faces
        '''
        # Convert BGR to RGB, opencv reads the image in BGR format
        pixels = img[...,::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = self.cascade_classifier.detectMultiScale(gray, 1.3, 5)
        faces = []
        facepoints = []
        for lm in landmarks:
            # Filter false faces
            x1, y1, w, h = lm
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            # Extract the region of the face
            face = pixels[y1:y2, x1:x2]
            if face.shape[0] > min_size[0] and face.shape[1] > min_size[1]:
                # Resize the result to the size expected for facenet
                image = Image.fromarray(face)
                image = image.resize((160, 160))
                array_img = np.asarray(image)
                faces.append(array_img)
                facepoints.append({'box':lm})
        return faces, facepoints

    
    def extract_faces_mtcnn(self, img, min_size=(100, 100)):
        ''' 
            Extract all the faces in the images.
            It uses the MTCNN detector to find the faces
        '''
        # Convert BGR to RGB, opencv reads the image in BGR format
        pixels = img[...,::-1]
        landmarks = self.face_detector.detect_faces(pixels)
        faces = []
        facepoints = []
        for lm in landmarks:
            # Filter false faces
            if lm['confidence'] >= 0.99:
                x1, y1, w, h = lm['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + w, y1 + h
                # Extract the region of the face
                face = pixels[y1:y2, x1:x2]
                if face.shape[0] > min_size[0] and face.shape[1] > min_size[1]:
                    # Resize the result to the size expected for facenet
                    image = Image.fromarray(face)
                    image = image.resize((160, 160))
                    array_img = np.asarray(image)
                    faces.append(array_img)
                    facepoints.append(lm)
            
        return faces, facepoints
        
    def normalize_vector(self, v):
        '''
            L2 normalization 
        '''
        return v / np.sqrt(np.sum(np.multiply(v, v)))

    def encode(self, faces):
        '''
            Encodes faces in a 128D vector
        '''
        if len(faces) == 0:
            return []
        # Scale pixel values
        mean, std = np.mean(faces), np.std(faces)
        scaled_pixels = (faces - mean) / std
        embeddings = self.facenet.predict(scaled_pixels)
        embeddings = self.normalize_vector(embeddings)
        return embeddings

class FaceClassifier(BaseFaceExtractor):
    '''
        Recognize people in images using a SVM classifier.
        The classifier is trained with embedding vectors using a pretrained facenet model
    '''
    def __init__(self, facenet_model, haarcascade_front_face_xml_path, names_path='', min_probability=0.75):
        self.facenet = facenet_model
        self.face_detector_cascade = cv2.CascadeClassifier(haarcascade_front_face_xml_path)
        self.temp_references = {}
        self.temp_names_count = 1
        self.min_probability = min_probability
        if names_path != '':
            with open(names_path, 'r') as file:
                self.names = file.read().splitlines()
        self.clf = SVC(kernel='linear', probability=True)
        super().__init__(self.facenet, self.face_detector_cascade)

    def train(self, training_images_path):
        self.names = os.listdir(training_images_path)
        print('Training for ', self.names)
        training_vectors = []
        training_names = []
        for idx, p in enumerate(self.names):
            # Ignore hidden files
            if p[0] != '.':
                print('Loading vector references for', p)
                person_path = training_images_path + '/' + p
                img_files = os.listdir(person_path)
                if '.DS_Store' in img_files:
                    img_files.remove('.DS_Store')
                for f in img_files:
                    img = cv2.imread(person_path + '/' + f)
                    face, _ = super().extract_faces_mtcnn(img)
                    # Checks if the face can be extracted
                    if len(face) > 0:
                        embeddings = super().encode(face)
                        training_vectors.append(embeddings[0])
                        training_names.append(idx)

        X_train = np.array(training_vectors)
        y_train = np.array(training_names)
        print('X_train.shape:', X_train.shape)
        print('y_train.shape:', y_train.shape)
        # Shuffle the data
        indexes = np.random.permutation(X_train.shape[0])
        X_train = X_train[indexes]
        y_train = y_train[indexes]
        # Train the model
        self.clf.fit(X_train, y_train)

    def recognize(self, img, graph):
        '''
            Find the most probably names of the people in the image. 
            When one person is not recognized it returns 'Unknown'
            Parameters
            ----------
            img: Opencv image
            face_detector: str
                - 'mtcnn' to use the MTCNN face detector. It has excellent accuracy but it's slow
                - 'opencv' to use the cascadeClassifier detector provided by OpenCV. Only recognize front faces but it's fast
            temp_names: Boolean
                Flag to assign temporal names to unknown people
            
            Returns
            -------
            names: List
            landmarks: List
                (x, y, w, h) points for each face in the image
            probs: List
                prediction confidence for each person classified
        '''
        names = []
        with graph.as_default():
            faces, landmarks = super().extract_faces_opencv(img)
        if len(faces) == 0:
            return [], [], []
        with graph.as_default():
            embeddings = super().encode(faces)
        predicts = self.clf.predict_proba(embeddings)
        probs = [p[np.argmax(p)] for p in predicts]
        classes = [np.argmax(p) for p in predicts]
        more_likely_names = [self.names[c] for c in classes]

        for prob, name in zip(probs, more_likely_names):
            if prob >= self.min_probability:
                names.append(name)
            else:
                names.append('Unknown')
        return names, landmarks, probs


    def save(self, path):
        # Save the trained model
        pickle.dump(self.clf, open('{}/face_classifier.sav'.format(path), 'wb'))
        # Delete the previous name if the file exists
        with open('{}/names.txt'.format(path), 'w') as file: pass
        # Save the names file
        with open('{}/names.txt'.format(path), 'a+') as file:
            for name in self.names:
                file.write(name)
                file.write('\n')
        print('Model saved at', path)

    def load_model(self, path):
        self.clf = pickle.load(open(path, 'rb'))
