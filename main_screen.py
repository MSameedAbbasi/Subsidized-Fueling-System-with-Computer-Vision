import PIL
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import *
import datetime

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

category_dict = {'0': 'Large', '1':'Medium', '2':'Small'}
price_dict = {'0': 'Rs.230/ltr', '1':'Rs.180/ltr', '2':'Rs.130/ltr'}

def show_frame():

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Subsidized Fueling System - SFS")

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Subsidized Fueling System - SFS", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            # ml model
            image_path = "SFS Captures/" + datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S") + ".jpg"
            cv2.imwrite(image_path, frame)
            #print("{} written!".format(img_name))
            category = predict(image_path)
            # price = '300'
            print(category)
            
    cam.release()

    cv2.destroyAllWindows()



def display(res):
    root = Tk()
    root.geometry("600x350")
    root.bind('<Escape>', lambda e: root.quit())
    root.title("SFS")
    root.configure(background="#121212")
    
    category_label = Label(root, height=2, width=10, text="Category", font=('Bebas Neue', 24, 'bold'), bg="#1F1F1F", fg="white", anchor="center")
    category_label.place(x=230, y=110, anchor="center")

    category_value = Label(root, height=2, width=10, text="", font=('Bebas Neue', 24, 'bold'), bg="#1F1F1F", fg="#cbfdfd", anchor="center")
    category_value.place(x=380, y=110, anchor="center")

    price_label = Label(root, height=2, width=10, text="Price", font=('Bebas Neue', 24, 'bold'), bg="#1F1F1F", fg="white", anchor="center")
    price_label.place(x=230, y=200, anchor="center")

    price_value = Label(root, height=2, width=10, text="", font=('Bebas Neue', 24, 'bold'), bg="#1F1F1F", fg="#ffe4b5", anchor="center")
    price_value.place(x=380, y=200, anchor="center")

    category_value.config(text=category_dict[str(res)])
    price_value.config(text=price_dict[str(res)])
    root.mainloop()


def predict(image_path):
    model = load_model('model_inceptionresnetv2.h5')
    img = image.load_img(image_path, target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    prediction = np.argmax(model.predict(img_data), axis=1)[0]
    category = prediction
    display(category)



# var.set("1")
# var2.set("2")
show_frame()
#root.mainloop()