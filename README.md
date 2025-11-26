#  Virtual House Painter (Python + SegFormer CMP Facade)

This project implements a **Python-based virtual house painting system** that can recolor exterior building walls using deep learning and classical computer vision. It uses a **SegFormer semantic segmentation model fine-tuned on the CMP Facade dataset** to identify façade regions and applies realistic color transformations in **LAB color space** to preserve shading and texture.

---

##  Features

- **Deep segmentation** of building façades using SegFormer (CMP dataset)
- **Accurate wall masking** with morphological cleanup and largest-component filtering
- **LAB-based recoloring** that preserves lighting, shadows, and surface texture
- **Soft-edge blending** for natural transitions between painted and unpainted areas
- **Command-line usage** — no GUI required

---

##  Method Overview

### 1. Semantic Segmentation (Deep Learning)
We load the model:
Xpitfire/segformer-finetuned-segments-cmp-facade


This model segments façade elements such as:
- façade (main wall)
- pillars
- molding
- cornice

Only these classes are used to generate a **binary wall mask**.

### 2. Mask Cleaning (OpenCV)
We refine segmentation using:
- morphological closing / opening
- largest connected component extraction
- optional bottom-cropping to avoid floor/pavement artifacts

### 3. LAB Color Recoloring
Recoloring is done in **CIE LAB** space:
- Lightness (L) is preserved → natural realism
- A/B chroma channels are shifted towards the target color
- Final result is blended with a blurred mask for soft edges

---

##  Project Structure
virtual-house-painter/
├── README.md
├── requirements.txt
├── main.py
├── src/
│ ├── segmentation.py
│ ├── recolor.py
│ └── utils.py
├── examples/
│ ├── input.jpg
│ └── output.jpg
└── models/


---

##  Installation

```bash
git clone <repo-url>
cd virtual-house-painter
pip install -r requirements.txt
```
##  Usage
`python main.py --image path/to/image.jpg --color "#013220"`

##  Model

 - SegFormer-B0 (fine-tuned on CMP Facade)
   `https://huggingface.co/Xpitfire/segformer-finetuned-segments-cmp-facade`


#### Disclaimer

Parts of this project were developed with the assistance of modern coding tools, including AI-based code generation and debugging aids.
