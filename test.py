from keras.models import load_model
import cv2
import numpy as np
import sys

filepath = sys.argv[1]

REV_CLASS_MAP = {
    0: "none",
    1: "paper",
    2: "rock",
    3: "scissors"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("model_stone_paper_scissors.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img/255
img = np.expand_dims(img, axis=0)

# predict the move made
pred = model.predict(img)
move = np.argmax(pred[0])
move_name = mapper(move)

print("Predicted: {}".format(move_name))
