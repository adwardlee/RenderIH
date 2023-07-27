from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

model_id = 'damo/cv_yolox-pai_hand-detection'
hand_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
img_path = 'videos/202301112.mp4'
cap = cv2.VideoCapture(img_path)
success, image = cap.read()
outshape = image.shape[:2]
out = cv2.VideoWriter('tmp_video.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 15,
                      outshape[::-1])
count = -1
while cap.isOpened():
    count += 1
    success, ori_image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    try:
        output = hand_detection(ori_image)
    except:
        #cv2.imwrite('tmp/tmp_{}.jpg'.format(count), ori_image)
        out.write(ori_image)
        continue
    for one in output['boxes']:
        x1, y1, x2, y2 = one
        cv2.rectangle(ori_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.imwrite('tmp/tmp_{}.jpg'.format(count), ori_image)
    out.write(ori_image)

    #print(output)
out.release()