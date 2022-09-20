import numpy as np
import tensorflow as tf

from typing import Union


IDX_CLASS = [
    "BACKGROUND",
    "ArticulatedTruck",
    "Bicycle",
    "Bus",
    "Car",
    "Motorcycle",
    "MotorizedVehicle",
    "NonMotorizedVehicle",
    "Pedestrian",
    "PickupTruck",
    "SingleUnitTruck",
    "WorkVan"
]

CLASS_IDX = dict([
    (class_label, idx) for (idx,class_label) in enumerate(IDX_CLASS)
])

OBJ_DET_FEATURE_DESCRIPTION = {
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/key/sha256": tf.io.FixedLenFeature([], tf.string),
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/format": tf.io.FixedLenFeature([], tf.string),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    "image/source_id": tf.io.FixedLenFeature([], tf.string),
}

def encode_image(image_tensor, type = "png"):
    if type == "png":
        return tf.io.encode_png(image=image_tensor)
    elif type == "jpeg":
        return tf.io.encode_jpeg(image=image_tensor)
    else:
        raise ValueError(f"{type} is not supported [`png`, `jpeg`]")

def decode_image(encoded_image: tf.Tensor) -> tf.Tensor:
    return tf.image.decode_image(
        encoded_image, channels=3, expand_animations=False
    )

class BoundingBox:
    def __init__(
        self,
        boxes: Union[list, tuple, np.ndarray, tf.Tensor],
        box_type: str = "corner"
    ):  
        assert box_type in ["corner", "center"], "box_type is not among the accepted types [`corner`, `center`]"
        self._box_type = box_type
        boxes = self._process_input_boxes(boxes = boxes)
        
        if self._box_type == "corner":
            self._corner_boxes = boxes
            self._center_boxes = self.to_center(self._corner_boxes)
        else:
            self._center_boxes = boxes
            self._corner_boxes = self.to_corner(self._center_boxes)
    
    def set_center_boxes(self, boxes):
        self._center_boxes = self._process_input_boxes(boxes)
        self._corner_boxes = self.to_corner(self._center_boxes)

    def set_corner_boxes(self, boxes):
        self._corner_boxes = self._process_input_boxes(boxes)
        self._center_boxes = self.to_center(self._corner_boxes)

    def _process_input_boxes(self, boxes):
        if isinstance(boxes, list) or isinstance(boxes, tuple) or (isinstance(boxes, np.ndarray) and len(boxes.shape) == 1):
            boxes = np.expand_dims(np.array(boxes), axis=0)
        elif isinstance(boxes, tf.Tensor):
            boxes = boxes.numpy()
        return boxes

    def _process_output_boxes(self, boxes, trim_boxes = True):
        boxes = boxes.squeeze()
        if trim_boxes:
            boxes = self.trim_boxes(boxes)
        if isinstance(boxes, np.ndarray) and (boxes.shape[0] == 1):
            return boxes.tolist()
        return boxes

    @property
    def length(self,):
        if self.is_center():
            return self._center_boxes.shape[0]
        return self._corner_boxes.shape[0]

    def scale_boxes(
        self,
        boxes: Union[list, tuple, np.ndarray],
        w: Union[float, int],
        h: Union[float, int]
    ):
        if isinstance(w, int): w = float(w)
        if isinstance(h, int): h = float(h)
        
        scaled = np.stack(
            [boxes[..., 0]*w, boxes[..., 1]*h, boxes[..., 2]*w, boxes[..., 3]*h],
            axis=-1,
        )
        return self._process_output_boxes(scaled.astype(np.int64), trim_boxes=False)

    def scale_center(
        self,
        w: Union[float, int],
        h: Union[float, int]
    ):
        return self.scale_boxes(self._center_boxes, w, h)

    def scale_corner(
        self,
        w: Union[float, int],
        h: Union[float, int]
    ):
        return self.scale_boxes(self._corner_boxes, w, h)

    @property
    def is_center(self,):
        return self._box_type == "center"

    @property
    def is_corner(self,):
        return self._box_type == "corner"

    @property
    def box_type(self,):
        return self._box_type

    @property
    def range(self):
        return (self._corner_boxes.min(), self._corner_boxes.max())

    def to_center(
        self,
        boxes: Union[list, tuple, np.ndarray]
    ):
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]

        return self._process_output_boxes(
            np.stack(
                [boxes[..., 0] + (w / 2), boxes[..., 1] + (h / 2), w, h],  # cx  # cy  # w  # h
                axis=-1,
            )
        )
    
    def to_corner(
        self,
        boxes: Union[list, tuple, np.ndarray]
    ):
        return self._process_output_boxes(
            np.stack(
                [
                    boxes[..., 0] - (boxes[..., 2] / 2),  # xmin
                    boxes[..., 1] - (boxes[..., 3] / 2),  # ymin
                    boxes[..., 0] + (boxes[..., 2] / 2),  # xmax
                    boxes[..., 1] + (boxes[..., 3] / 2),  # ymax
                ], axis=-1,
            )
        )

    @property
    def corner_boxes(self):
        return self._process_output_boxes(self._corner_boxes)

    @property
    def center_boxes(self):
        return self._process_output_boxes(self._center_boxes)
    
    def trim_boxes(self, boxes: np.ndarray, w = None, h = None):

        if not w and not h:
            w = 1.0
            h = 1.0

        boxes[..., 0] = np.clip(boxes[..., 0], a_min=0.0, a_max=w)
        boxes[..., 1] = np.clip(boxes[..., 1], a_min=0.0, a_max=h)
        boxes[..., 2] = np.clip(boxes[..., 2], a_min=0.0, a_max=w)
        boxes[..., 3] = np.clip(boxes[..., 3], a_min=0.0, a_max=h)
        return boxes

    def proportional_change(self, *, boxes=None, proportion=0.1, box_type = "corner"):
        boxes = self._process_input_boxes(boxes=boxes)

        if boxes is None:
            boxes = self._center_boxes
        elif boxes and box_type == "corner":
            boxes = self.to_center(boxes=boxes)

        boxes = boxes.copy()
        adj_boxes = np.stack([
                boxes[..., 0],
                boxes[..., 1],
                boxes[..., 2]*(1+proportion),
                boxes[..., 3]*(1+proportion)
            ],
            axis = -1
        )

        if box_type == "corner":
            adj_boxes =  self.to_corner(boxes=adj_boxes)

        return self._process_output_boxes(adj_boxes)
        
    def shift_center(self, *, boxes = None, x_shift_by: float, y_shift_by: float):
        if boxes:
            boxes = self._process_input_boxes(boxes=boxes)
        else:
            boxes = self._center_boxes.copy()

        boxes[..., 0] *= (1+x_shift_by)
        boxes[..., 1] *= (1+y_shift_by)
        return self._process_output_boxes(boxes=boxes)
    
    def shift_corner(self, *, boxes = None, x_shift_by: float, y_shift_by: float):
        if boxes:
            boxes = self._process_input_boxes(boxes=boxes)
        else:
            boxes = self._corner_boxes.copy()
            
        boxes[..., 0] *= (1+x_shift_by)
        boxes[..., 1] *= (1+y_shift_by)
        boxes[..., 2] *= (1+x_shift_by)
        boxes[..., 3] *= (1+y_shift_by)
        return self._process_output_boxes(boxes=boxes)

    def shift_boxes(self, x_shift_by, y_shift_by):
        if self.is_corner():
            return self.shift_corner(x_shift_by=x_shift_by, y_shift_by=y_shift_by)
        return self.shift_center(x_shift_by=x_shift_by, y_shift_by=y_shift_by)

    def __call__(self):
        if self.is_corner():
            return self.corner_boxes()
        return self.center_boxes()

class RecordData:
    def __init__(
        self,
        *,
        image,
        boxes,
        labels,
        height,
        width,
        sha256,
        format,
        text,
        source_id
    ):
        self.image = image
        self.boxes = boxes
        self.labels = labels
        self.width = width
        self.height = height
        self.sha256 = sha256
        self.format = format
        self.text = text
        self.source_id = source_id

    @property
    def is_empty(self, ):
        pass

class RecordBuffer:
    def __init__(self, records_list = list(), min_length = 0):
        self.buffer = records_list
        self.min_buffer_length = min_length

    @property
    def is_empty(self):
        return len(self.buffer) == 0

    def __len__(self, ):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    @property
    def images(self,):
        return [record.image for record in self.buffer]
    
    @property
    def boxes(self,):
        return [record.boxes for record in self.buffer]
    
    @property
    def labels(self,):
        return [record.labels for record in self.buffer]

    @property
    def sha256s(self,):
        return [record.sha256 for record in self.buffer]

    @property
    def formats(self,):
        return [record.format for record in self.buffer]

    @property
    def source_ids(self,):
        return [record.source_id for record in self.buffer]

    @property
    def heights(self,):
        return [record.height for record in self.buffer]

    @property
    def widths(self,):
        return [record.width for record in self.buffer]

    @property
    def image_sizes(self,):
        return np.array([(record.width.numpy(), record.height.numpy()) for record in self.buffer])

    def empty_buffer(self,):
        self.buffer = list()

    def add(self, record):
        if not self.is_empty and len(self) >= self.min_buffer_length:
            self.buffer.pop(0)
        self.buffer.append(record)
    
def parse_record(example: tf.Tensor):
    parsed_example = tf.io.parse_single_example(
        example, features=OBJ_DET_FEATURE_DESCRIPTION
    )

    image = decode_image(parsed_example["image/encoded"]).numpy()
    height = parsed_example["image/height"].numpy()
    width = parsed_example["image/width"].numpy()
    sha256 = parsed_example["image/key/sha256"]
    format = parsed_example["image/format"]
    text = parsed_example["image/object/class/text"]
    source_id = parsed_example["image/source_id"]
    class_labels = parsed_example["image/object/class/label"]

    boxes = np.array([[xmin, ymin, (xmax-xmin), (ymax-ymin)] for xmin, ymin, xmax, ymax in zip(
        tf.sparse.to_dense(parsed_example["image/object/bbox/xmin"]).numpy(),
        tf.sparse.to_dense(parsed_example["image/object/bbox/ymin"]).numpy(),
        tf.sparse.to_dense(parsed_example["image/object/bbox/xmax"]).numpy(),
        tf.sparse.to_dense(parsed_example["image/object/bbox/ymax"]).numpy(),
    )])

    labels = tf.sparse.to_dense(class_labels).numpy()
    text = tf.sparse.to_dense(text)
    return RecordData(
        image=image,
        boxes=boxes,
        labels=labels,
        height=height,
        width=width,
        sha256=sha256,
        format=format,
        text=text,
        source_id=source_id
    )
