from PIL import Image
import numpy as np
import requests
import json
import time


image = Image.open('one.png')
iamge = image.resize((28, 28))
# image = image.tolist()

image = image.resize((28, 28))

image.save('onee.png')
image = np.asarray(image)

image = image.tolist()
# print(image)
data = {'image': image}
print(data)
URL = 'http://127.0.0.1:5000/predict'

# time start
start_time = time.time()
result = requests.post(URL, json.dumps(data))
print(f"Prediction = {result.text}")
#  time end
end_time = time.time()
print(f"Time taken for prediction = {end_time - start_time}")
