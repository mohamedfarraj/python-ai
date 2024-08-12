import cv2

# Mean values used for model normalization
MODEL_MEAN_VALUES = (78.4463377603, 87.7689143744, 114.895847746)

# Age and gender categories
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_models():
    try:
        age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt', 'data/gender_net.caffemodel')
        return age_net, gender_net
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def predict_age_and_gender(image_path, age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or unable to load.")
        return

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        print("No faces found")
        return

    print(f"Found {len(faces)} face(s)")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        face_img = image[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender: " + gender)

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age: " + age)

        label = f"{gender}, {age}"
        cv2.putText(image, label, (x, y - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Age and Gender Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    age_net, gender_net = load_models()
    if age_net is not None and gender_net is not None:
        predict_age_and_gender('images/girl1.jpg', age_net, gender_net)
