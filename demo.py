# /usr/bin/python3
import cv2
import numpy as np
import sys
import tensorflow as tf

from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def specify(parameter, img):
  img_width = len(img[0])
  img_height = len(img)
  for i in range(img_height):
    for j in range(parameter[0]):
      img[i][j][0] = 0
      img[i][j][1] = 0
      img[i][j][2] = 0
  for i in range(img_height):
    for j in range(parameter[0] + parameter[2], img_width):
      img[i][j][0] = 0
      img[i][j][1] = 0
      img[i][j][2] = 0
  for i in range(parameter[1]):
    for j in range(img_width):
      img[i][j][0] = 0
      img[i][j][1] = 0
      img[i][j][2] = 0
  for i in range(parameter[1] + parameter[3], img_height):
    for j in range(img_width):
      img[i][j][0] = 0
      img[i][j][1] = 0
      img[i][j][2] = 0
def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor

def face_dect(image):
  """
  Detecting faces in image
  :param image: 
  :return:  the coordinate of max face
  """
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    return None
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return face_image

def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image

def draw_emotion():
  pass

def demo(modelPath, showBox=False):
  face_x = tf.compat.v1.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(face_x)
  probs = tf.nn.softmax(y_conv)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)
  sess = tf.Session()
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

  video_path = "test.mp4"
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
  size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频的大小

  fourcc = cv2.VideoWriter_fourcc(*'mpeg')  # 要保存的视频格式
  # 把处理过的视频保存下来
  output_viedo = cv2.VideoWriter()
  # 保存的视频地址
  video_save_path = 'trans.mp4'
  output_viedo.open(video_save_path, fourcc, fps, size, True)
  result = None

  while True:
    ret, frame = cap.read()
    detected_face, face_coor = format_image(frame)
    if showBox:
      if face_coor is not None:
        specify(face_coor,frame)
    if detected_face is not None:
      tensor = image_to_tensor(detected_face)
      result = sess.run(probs, feed_dict={face_x: tensor})
      print(result)
    output_viedo.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
'''
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                      (255, 0, 0), -1)
'''


