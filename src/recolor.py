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

def recolor_walls(image_pil: Image.Image, wall_mask: np.ndarray, target_hex: str):
  img_rgb= np.array(image_pil).astype(np.uint8)
  h,w,_=img_rgb.shape

  wall_mask= wall_mask.astype(np.uint8)
  if wall_mask.shape !=(h,w):
    wall_mask=cv2.resize(wall_mask, (w,h), interpolation=cv2.INTER_NEAREST)

  img_bgr= cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  img_lab= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
  L,A,B= cv2.split(img_lab)
  wall_indices= wall_mask==1
  if wall_indices.sum()==0:
    print("NO WALL PIXELS DETECTED")
    return image_pil

  avg_A= A[wall_indices].mean()
  avg_B= B[wall_indices].mean()

  target_rgb = np.array([[hex_to_rgb(target_hex)]],dtype=np.uint8)
  target_bgr= cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
  target_lab= cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0,0]
  tL, tA, tB= target_lab

  DA=tA-avg_A
  DB=tB-avg_B

  A_new= A.astype(np.float32)
  B_new= B.astype(np.float32)
  A_new[wall_indices]+=DA
  B_new[wall_indices]+=DB

  A_new = np.clip(A_new,0,255).astype(np.uint8)
  B_new = np.clip(B_new,0,255).astype(np.uint8)

  lab_new = cv2.merge([L, A_new, B_new])
  bgr_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
  rgb_new = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2RGB)

  mask_f= wall_mask.astype(np.float32)
  mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)
  mask_f = mask_f[:, :, None]
  blended = img_rgb.astype(np.float32) * (1 - mask_f) + rgb_new.astype(np.float32) * mask_f
  blended = np.clip(blended, 0, 255).astype(np.uint8)

  return Image.fromarray(blended)


def overlay_mask(image_pil: Image.Image, wall_mask: np.ndarray):
  img= np.array(image_pil).astype(np.uint8)
  h,w,_= img.shape
  mask = wall_mask
  if mask.shape !=(h,w):
    mask=cv2.resize(mask, (w,h), interpolation= cv2.INTER_NEAREST)
  mask = mask.astype(bool)

  overlay= img.copy()
  overlay[mask]= [255,0,0]

  vis = cv2.addWeighted(img, 0.6,overlay, 0.8, 0)
  return Image.fromarray(vis)

print("Upload an image:")
uploaded= files.upload()

img_name=list(uploaded.keys())[0]
image=Image.open(img_name).convert("RGB")
target_hex= "#013220"

wall_mask= wall_mask(image, keep_top_k=1)
painted = recolor_walls(image, wall_mask, target_hex)
overlay = overlay_mask(image,wall_mask)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Recolored {target_hex}")
plt.imshow(painted)
plt.axis("off")

plt.show()
