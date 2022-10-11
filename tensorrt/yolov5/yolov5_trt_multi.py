"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import imp
import multiprocessing
import os
import queue
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed


CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
# global isSaveImage
isSaveImage=True

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def get_img_path_batches_by_splitlist(batch_size, img_path_list):
    ret = []
    batch = []
    for name in img_path_list:
        if len(batch) == batch_size:
            ret.append(batch)
            batch = []
        batch.append(name)
    if len(batch) > 0:
        ret.append(batch)
    return ret

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """
    """ ctx, """
    def __init__(self, engine_file_path, output_save_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print(binding)
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print('bingding:', binding, engine.get_binding_shape(binding),',dtype',dtype)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings. GPU 地址
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store 保存trt相关的对象：stream\context\engine\in_outputs \bindings\batch_size
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.save_path = output_save_path

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess 组装batch数据
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer batch数据copy到GPU
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            if isSaveImage:
                # Draw rectangles and labels on the original image # 测试时间，不画图
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    plot_one_box(
                        box,
                        batch_image_raw[i],
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[j])], result_scores[j]
                        ),
                    )
        return batch_image_raw, end - start
    
    def infer_single(self, input_image,image_raw_path,origin_h, origin_w):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        # self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess 组装batch数据
        batch_image_raw_path = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        
        for i in range(self.batch_size):
            batch_image_raw_path.append(image_raw_path)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer batch数据copy到GPU
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        # self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        batch_image_raw=[]
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            if isSaveImage:
                # Draw rectangles and labels on the original image # 测试时间，不画图
                image_raw = cv2.imread(batch_image_raw_path[i])
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    plot_one_box(
                        box,
                        image_raw,
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[j])], result_scores[j]
                        ),
                    )
                batch_image_raw.append(image_raw)
        return batch_image_raw, end - start

    def infer_thread(self, input_image,image_raw, origin_w, origin_h):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess 组装batch数据
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        
        for i in range(self.batch_size):
            # input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer batch数据copy到GPU
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            if isSaveImage:
                # Draw rectangles and labels on the original image # 测试时间，不画图
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    plot_one_box(
                        box,
                        batch_image_raw[i],
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[j])], result_scores[j]
                        ),
                    )
        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        if isSaveImage:
            for i, img_path in enumerate(self.image_path_batch):    # 测试时间，不保存图
                parent, filename = os.path.split(img_path)
                save_name = os.path.join(self.yolov5_wrapper.save_path, filename)
                # Save image
                cv2.imwrite(save_name, batch_image_raw[i])
            print('input->{}, time->{:.2f}ms, saving into {}/'.format(self.image_path_batch, use_time * 1000, save_name)) 

class inferThread_2(threading.Thread):
    def __init__(self, pool_index, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.pool_idex = pool_index

    def run(self):
        print(f"in infer thread,qsize={queue_data.qsize()}")
        while True:
            if queue_data.empty():
                time.sleep(2)
                if queue_data.empty():
                    break
                continue
            # print(f"pool_idex:{self.pool_idex}, pop one, remaining size: {queue_data.qsize()}")
            data_node = queue_data.get()
            image = data_node["image"]
            image_path = data_node["image_path"]
            image_raw = data_node["image_raw"]
            ori_w = data_node["w"]
            ori_h = data_node["h"]
            # print(f">>image_path:{image_path},w={ori_w},h={ori_h}")
            batch_image_raw, use_time = self.yolov5_wrapper.infer_thread(image, image_raw, ori_w, ori_h)
            if isSaveImage:
                parent, filename = os.path.split(image_path)
                save_name = os.path.join(self.yolov5_wrapper.save_path, filename)
                # Save image
                cv2.imwrite(save_name, batch_image_raw[0])
                print('input->{}, time->{:.2f}ms, saving into {}/'.format(image_path, use_time * 1000, save_name)) 


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))

def yolov5_demo():
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov5s.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels
    global categories
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    save_path = 'output/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path,save_path)

    # try 部分改为多线程处理
    try:
        print('batch size is', yolov5_wrapper.batch_size)
        # 5000张
        image_dir = "/workspace/project/dataset/benchmarch_data/detection/val2017"  #"images/"
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        
        start_ = time.time()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, batch)
            thread1.start()
            thread1.join()
        end_ = time.time()
        print(f"time==>{(end_ - start_) * 1000:.2f}")
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()


def get_img_path_by_processor_number(nProcessor, batch_size, img_dir):
    """
        每个进程平均处理图片数量
        return [[100],[100],[100]...[100]]
    """
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        files=files[0:2000]
        need_files_num = len(files) // (nProcessor * batch_size) * (nProcessor * batch_size)
        num_per_prossor = need_files_num // nProcessor
        print(f">>>>>>>>>>>>>>>>>> acctualy need {need_files_num} files")
        for name in files[:need_files_num]:
            if len(batch) == num_per_prossor:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

# class data

def process_circlely(pool_index,yolov5_wrapper, data_path_list):
    """ 
        原处理方式：同一个线程内部，循环处理没张图
    """
    # try 部分改为多线程处理
    try:
        print('batch size is', yolov5_wrapper.batch_size)

        # image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, data_path_list)              # 根据engine的batchsize 
        image_path_batches = get_img_path_batches_by_splitlist(yolov5_wrapper.batch_size, data_path_list)
        
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        
        # 迭代batch数据，在一个线程中，循环处理
        start_ = time.time()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, batch)
            thread1.start()
            thread1.join()
        end_ = time.time()
        print(f"processor index : {pool_index}, time==>{(end_ - start_) * 1000:.2f} ms")

        # 迭代batch数据，dataloader

    finally:
        print('release resource')
        # destroy the instance
        yolov5_wrapper.destroy()

import torch.utils.data as data
def process_mutil_loaddata(pool_index,yolov5_wrapper, data_path_list, data_batch_size=4, thread_nums=0):
    
    class dataset(data.Dataset):

        def __init__(self, data_path_list,input_w,input_h) -> None:
            super().__init__()
            self.data_path = data_path_list
            self.input_w = input_w
            self.input_h = input_h

        def __len__(self):
            return len(self.data_path)

        def preprocess_image(self, raw_bgr_image):
            """
            description: Convert BGR image to RGB,
                        resize and pad it to target size, normalize to [0,1],
                        transform to NCHW format.
            param:
                input_image_path: str, image path
            return:
                image:  the processed image
                image_raw: the original image
                h: original height
                w: original width
            """
            image_raw = raw_bgr_image
            h, w, c = image_raw.shape
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            # Calculate widht and height and paddings
            r_w = self.input_w / w
            r_h = self.input_h / h
            if r_h > r_w:
                tw = self.input_w
                th = int(r_w * h)
                tx1 = tx2 = 0
                ty1 = int((self.input_h - th) / 2)
                ty2 = self.input_h - th - ty1
            else:
                tw = int(r_h * w)
                th = self.input_h
                tx1 = int((self.input_w - tw) / 2)
                tx2 = self.input_w - tw - tx1
                ty1 = ty2 = 0
            # Resize the image with long side while maintaining ratio
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0
            # HWC to CHW format:
            image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            # image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order, also known as "C order":
            image = np.ascontiguousarray(image)
            return image , h, w #image_raw, 
        def _preprocess(self, index):
            # batch_input_image = np.empty(shape=[1, 3, self.input_h, self.input_w])
            # for i, image_raw in enumerate(raw_image_generator):
            image_path = self.data_path[index]
            image_raw = cv2.imread(image_path)
            input_image,  origin_h, origin_w  = self.preprocess_image(image_raw)   #image_raw,
            # batch_image_raw.append(image_raw)
            # batch_origin_h.append(origin_h)
            # batch_origin_w.append(origin_w)
            # np.copyto(batch_input_image[i], input_image)   

            return input_image,image_path, origin_h, origin_w #[image_raw],

        def __getitem__(self, index):
            return self._preprocess(index) 


    input_w = yolov5_wrapper.input_w
    input_h = yolov5_wrapper.input_h

    mydata = dataset(data_path_list,input_w,input_h)
    dataloader = data.DataLoader(mydata,batch_size=data_batch_size,num_workers=thread_nums) #, pin_memory=True,drop_last=True
    
    try:
        start_ = time.time()
        for input_image,image_paths, origin_h, origin_w  in dataloader: # 
            input_image = input_image.numpy()
            origin_h = origin_h.numpy()
            origin_w = origin_w.numpy()

            for i in range(data_batch_size):
                image_raw_d, use_time = yolov5_wrapper.infer_single(input_image[i,:],image_paths[i],origin_h[i], origin_w[i])
                if isSaveImage:
                    parent, filename = os.path.split(image_paths[i])
                    save_name = os.path.join(yolov5_wrapper.save_path, filename)
                    # Save image
                    cv2.imwrite(save_name, image_raw_d[0]) 
                    print('input->{}, time->{:.2f}ms, saving into {}/'.format(image_paths[i], use_time * 1000, save_name))
        end_ = time.time()
        print(f"processor index : {pool_index}, time==>{(end_ - start_) * 1000:.2f} ms")

    finally:
        print('release resource')
        yolov5_wrapper.destroy()


def Processor_func(data_path_list, thread_nums, pool_index, num_load=4):
    """
        进程函数
    """
    def productor_loaddata_to_queue(pool_index, thread_id, data_list,
                    input_w=640,input_h=640):
        print(f"pool_index:{pool_index},thread_id:{thread_id}, data_list type:{type(data_list)},len={len(data_list)}")
        for index in range(len(data_list)):
            image_path = data_list[index]
            # print(f"productor: {image_path}")
            image_raw = cv2.imread(image_path)

            h, w, c = image_raw.shape
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            # Calculate widht and height and paddings
            r_w = input_w / w
            r_h = input_h / h
            if r_h > r_w:
                tw = input_w
                th = int(r_w * h)
                tx1 = tx2 = 0
                ty1 = int((input_h - th) / 2)
                ty2 = input_h - th - ty1
            else:
                tw = int(r_h * w)
                th = input_h
                tx1 = int((input_w - tw) / 2)
                tx2 = input_w - tw - tx1
                ty1 = ty2 = 0
            # Resize the image with long side while maintaining ratio
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0
            # HWC to CHW format:
            image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            # image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order, also known as "C order":
            image = np.ascontiguousarray(image)
            
            dat_node = dict()
            dat_node["image"]=image
            dat_node["image_path"]=image_path
            dat_node["image_raw"]=image_raw
            dat_node["w"]=w
            dat_node["h"]=h

            queue_data.put(dat_node)
            # print(f"pool_index:{pool_index}, thread_id:{thread_id}, put into queue, size={queue_data.qsize()}")
            # return image , h, w #image_raw, 

    
    print(f"processor args, data_path_list:{len(data_path_list)}, thread_nums:{thread_nums},pool_index:{pool_index}")
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov5s.engine"

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels
    global categories
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    save_path = 'output_'+ str(pool_index) +'/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print(f"make output dir {save_path}")
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path, save_path)
    input_w = yolov5_wrapper.input_w
    input_h = yolov5_wrapper.input_h

    # # *************************************************
    # # 两种多进程方式：1
    # process_circlely(pool_index,yolov5_wrapper, data_path_list)
    # # 方式：2
    # process_mutil_loaddata(pool_index, yolov5_wrapper, data_path_list, num_load, thread_nums)

    # 方式3 生产者消费者模式： 队列 + 多线程取数据 + 单线程处理
    global queue_data
    queue_data = queue.Queue(128)
    # productor
    start_ = time.time()
    t_list=[]
    num_per_thread = len(data_path_list) // thread_nums
    thread_id = 0
    for batch_i in range(0, len(data_path_list), num_per_thread):
        data_list = data_path_list[batch_i: batch_i+num_per_thread]
        t = threading.Thread(target=productor_loaddata_to_queue, args=(pool_index, thread_id, data_list,input_w,input_h,))
        thread_id+=1
        t_list.append(t)
        t.start()
    # comsummer
    infer_t = inferThread_2(pool_index, yolov5_wrapper)
    infer_t.start()
    # waiting
    for rt in t_list:
        rt.join()
    infer_t.join()
    end_ = time.time()
    print(f"processor index : {pool_index}, time==>{(end_ - start_) * 1000:.2f} ms")
    yolov5_wrapper.destroy()
    

def multiProcessor_and_multiThread():
    """
        多进程套多线程
    """
    
    thread_num = 4              # 线程最大数量  {dataloader 多线程取数据有问题！！}
    pool_num = 2                # 进程数量
    number_load_by_mutilT = 4   # 多线程取数据数
    pool_count = 0

    # 5000张
    image_dir = "/workspace/project/dataset/benchmarch_data/detection/val2017"
    image_path_batches = get_img_path_by_processor_number(pool_num, number_load_by_mutilT, image_dir)   # 确保每个进程的处理数量一致，且能被number_load_by_mutilT整除
    print("number batch data:", len(image_path_batches))


    p_list = []
    for data_list in image_path_batches:
        proc = multiprocessing.Process(target=Processor_func, args=(data_list, thread_num, pool_count, number_load_by_mutilT))
        pool_count += 1
        proc.start()
        # time.sleep(2)
        p_list.append(proc)
    for res in p_list:
        res.join()


if __name__ == "__main__":
    # yolov5_demo()
    multiprocessing.set_start_method('spawn')
    multiProcessor_and_multiThread()
    

    

    
    
