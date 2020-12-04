
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from model import VPNet


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def cal_fps_from_video(model, device, live_cam=True, video_path=None, show_video=True, record_video=False,
                       output_prefix=''):
    if record_video:
        output_prefix += "-Infer"

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    fontColor = (255, 255, 255)
    lineType = 2

    size = (1024, 512)
    if record_video:
        out = cv2.VideoWriter(f"{output_prefix}-{video_path}", cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    fps_list = []

    model.eval()
    mean_list = {'x': [0.5 for _ in range(6)], 'y': [0.5 for _ in range(6)]}
    if live_cam:
        vidcap = open_cam_onboard(2048, 1024)
    else:
        vidcap = cv2.VideoCapture(video_path)

    while True:

        start_time = time.time()
        try:
            _, img = vidcap.read()
        #             if not live_cam:
        #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            break
        time_accum = time.time() - start_time

        try:
            if show_video:
                height, width = img.shape[:2]
                source_size = (width, height)
        except:
            break

        im_pil = Image.fromarray(img)

        start_time = time.time()
        inputs = transform(im_pil)[np.newaxis, ...]
        time_accum += (time.time() - start_time)

        # Send data to GPU
        inputs = inputs.to(device, dtype=torch.float32)

        start_time = time.time()  # start time of the loop
        outputs = model(inputs)
        time_accum += (time.time() - start_time)

        fps = 1.0 / time_accum  # FPS = 1 / time to process loop
        fps_list.append(fps)
        # clear_output(wait=True)
        print("FPS: ", round(fps, 2))

        if show_video:
            (x, y) = outputs[0]  # [0], outputs[0][1]

            x, y = x.item(), y.item()
            if x < 0.4:
                x = 0.4
            elif x > 0.6:
                x = 0.6
            #         if y < 0.4:
            #             y = 0.4
            #         elif y > 0.6:
            #               y = 0.6
            #         y -= 0.05
            mean_list['x'].pop(0)
            mean_list['x'].append(x)
            mean_list['y'].pop(0)
            mean_list['y'].append(y)
            x = np.mean(mean_list['x'])
            y = np.mean(mean_list['y'])

            pred_cir_pos = (int(x * source_size[0]), int(y * source_size[1]))

            temp = img
            cv2.circle(temp, pred_cir_pos, 5, (244, 35, 232), 10)  # Pink Prediction

            temp = cv2.resize(temp, size)
            if record_video:
                out.write(temp)

            cv2.imshow('Inference', temp)
            key = cv2.waitKey(10)
            if key == 27:  # ESC key: quit program
                cv2.destroyAllWindows()
                if record_video:
                    out.release()
                return f"Mean FPS: {int(np.mean(fps_list[1:]))} \
                         Min FPS: {int(np.min(fps_list[1:]))}  \
                         Max FPS: {int(np.max(fps_list[1:]))}"

    cv2.destroyAllWindows()
    if record_video:
        out.release()
    return f"Mean FPS: {int(np.mean(fps_list[1:]))} \
             Min FPS: {int(np.min(fps_list[1:]))}  \
             Max FPS: {int(np.max(fps_list[1:]))}"


transform = transforms.Compose([transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

# PATH = './better_scheduler_nesterov-1e-3-04-24-19.pkl'
PATH = './coarse_train_dataset_le-3_04-28-19.pkl'


model = VPNet()
model.load_state_dict(torch.load(PATH, map_location='cuda:0'))

print(torch.cuda.get_device_name(0))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device, dtype=torch.float32)


video_path = 'munich.mp4'

print(cal_fps_from_video(model, device, live_cam=False, video_path=video_path,
                         show_video=True, record_video=False, output_prefix='pred'))
