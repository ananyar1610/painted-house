import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from google.colab import files
import matplotlib.pyplot as plt

def hex_to_rgb(hex_color: str):
  hex_color= hex_color.lstrip("#")
  x=tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))
  return x

def load_cmp_facade_model():
  model_name= 'Xpitfire/segformer-finetuned-segments-cmp-facade'
  processor = SegformerImageProcessor.from_pretrained(model_name)
  model = SegformerForSemanticSegmentation.from_pretrained(model_name)
  model.eval()
  return processor, model
processor, model= load_cmp_facade_model()


def wall_mask( image_pil: Image.Image, keep_top_k: int = 1):
  inputs= processor(images=image_pil, return_tensors='pt')
  with torch.no_grad():
    outputs=model(**inputs)
    target_size= [image_pil.size[::-1]]

    seg_maps= processor.post_process_semantic_segmentation(outputs, target_sizes=target_size)
    seg=seg_maps[0].cpu().numpy()
    id2label=model.config.id2label

    GOOD_KEYWORDS=['facade', 'cornice', 'pillar', 'molding']
    wall_ids=[]
    for i, name in id2label.items():
      name_lower= name.lower()
      if any(k in name_lower for k in GOOD_KEYWORDS):
        wall_ids.append(int(i))

    if not wall_ids:
      unique, counts= np.unique(seg, return_counts=True)
      main_id = unique [np.argmax(counts)]
      raw_mask = (seg== main_id).astype(np.uint8)
    else:
      raw_mask = np.isin(seg, wall_ids).astype(np.uint8)

    kernel= np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels_cc= cv2.connectedComponents(mask.astype(np.uint8))

    if num_labels<=1:
      final_mask= mask
    else:
      areas=[]
      for lbl in range(1, num_labels):
        areas.append((lbl, (labels_cc==lbl).sum()))
      areas.sort(key=lambda x:x[1], reverse=True)
      keep_labels= [lbl for lbl, _ in areas[:keep_top_k]]
      final_mask= np.isin(labels_cc, keep_labels).astype(np.uint8)


    final_mask= cv2.dilate(final_mask, kernel, iterations=1)
    return final_mask
