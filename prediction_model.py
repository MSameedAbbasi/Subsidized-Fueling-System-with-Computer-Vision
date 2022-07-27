import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def predict(image_path):
    model = load_model('model_inceptionresnetv2.h5')
    img = image.load_img(image_path, target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    prediction = np.argmax(model.predict(img_data), axis=1)
    return prediction

# cam = cv2.VideoCapture(0)

# cv2.namedWindow("test")

# img_counter = 0

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#             # SPACE pressed
#             img_name = "opencv_frame_{}.png".format(img_counter)
#             cv2.imwrite(img_name, frame)
#             print("{} written!".format(img_name))
#             img_counter += 1

# cam.release()

# cv2.destroyAllWindows()