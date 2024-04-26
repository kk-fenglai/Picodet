import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 iou_threshold=0.3):
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (self.net.get_inputs()[0].shape[2],
                            self.net.get_inputs()[0].shape[3])

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] /
                                                      hw_scale)
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] *
                                 hw_scale), self.input_shape[1]
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0)
        else:
            img = cv2.resize(
                srcimg, self.input_shape, interpolation=cv2.INTER_AREA)

        return img, scale_factor

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def detect(self, srcimg):
        img, scale_factor = self.resize_image(srcimg)
        img = self._normalize(img)

        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {
            self.net.get_inputs()[0].name: blob,
            self.net.get_inputs()[1].name: scale_factor
        })

        outs = np.array(outs[0])
        expect_boxes = (outs[:, 1] > 0.5) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]

        # 存储目标框坐标信息的列表
        boxes_info = []

        for i in range(np_boxes.shape[0]):
            classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
            xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[
                i, 3]), int(np_boxes[i, 4]), int(np_boxes[i, 5])

            # 将目标框的坐标信息添加到列表中
            boxes_info.append({
                'class_id': classid,
                'confidence': conf,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

            # 在图像上绘制检测框和标签
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(
                srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
            label = f"{self.classes[classid]}: {conf:.2f}"
            cv2.putText(
                srcimg, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, thickness=2)

        return srcimg, boxes_info



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modelpath',
        type=str,
        default=r'C:\Users\lenovo\Desktop\picodet\picodet_l_320_lcnet_postprocessed.onnx',#替换成自己路径
        help="onnx filepath")
    parser.add_argument(
        '--classfile',
        type=str,
        default=r'C:\Users\lenovo\Desktop\picodet\coco_label.txt',#替换成自己路径
        help="classname filepath")
    parser.add_argument(
        '--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument(
        '--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()

    net = PicoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.confThreshold,
        iou_threshold=args.nmsThreshold)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, boxes_info = net.detect(frame)

        # 遍历目标框的坐标信息列表，并进行处理或保存
        for box_info in boxes_info:
            class_id = box_info['class_id']
            confidence = box_info['confidence']
            xmin, ymin = box_info['xmin'], box_info['ymin']
            xmax, ymax = box_info['xmax'], box_info['ymax']

            # 在控制台打印目标框的坐标信息
            print(f"Class ID: {class_id}, Confidence: {confidence:.2f}, "
                  f"Bounding Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

            # 进一步处理目标框的坐标信息，例如保存到文件中

        cv2.imshow('Object Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
