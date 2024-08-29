import warnings

import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
from PIL import Image

from .face_alignment import align
from .inference import to_input

warnings.filterwarnings("ignore")

SIZE = (112, 112)


class AdaFace:
    def __init__(self, onnx_path: str) -> None:
        self.onnx_path = onnx_path
        self.ort_session = ort.InferenceSession(self.onnx_path)

    def run(self, cropped_face_images: list[Image.Image]):
        pass

    def extract_features(self, cropped_face_images: list[Image.Image]) -> np.ndarray:
        """顔画像から特徴量を抽出する
        Args:
            cropped_face_images (list[Image.Image]): 顔部分を切り抜いた画像のリスト
        Returns:
            np.ndarray: 顔画像から抽出した特徴量 (n_faces, 512)
        """
        cropped_face_images: list[Image.Image] = [im.resize(SIZE) for im in cropped_face_images]
        cropped_faces: list[np.ndarray] = [to_input(face_pil_img) for face_pil_img in cropped_face_images]
        input_data = np.concatenate(cropped_faces)

        input_name = self.ort_session.get_inputs()[0].name
        features = self.ort_session.run(None, {input_name: input_data})
        features: np.ndarray = features[0]
        return features

    def crop_face(self, image_path: str) -> Image.Image:
        """image_pathから顔を切り取り、112x112のサイズにリサイズして返す"""
        face_pil_img = align.get_aligned_face(image_path)
        return face_pil_img
