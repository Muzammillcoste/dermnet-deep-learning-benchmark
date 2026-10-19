---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- image-classification
- vision-transformer
- dermatology
- medical-imaging
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: finetuned-dermnet
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: DermNet (23 classes)
      type: imagefolder
    metrics:
    - type: accuracy
      value: 0.7099
      name: Validation accuracy
    - type: loss
      value: 1.1935
      name: Validation loss
---

# finetuned-dermnet: Skin Disease Image Classifier (ViT)

A Vision Transformer (ViT-Base/16) fine-tuned to sort photos of skin conditions into **23 broad disease categories** from the DermNet image collection.

**Validation accuracy: 71.0%** across 23 classes (random guessing gets 4.3%).

> [!WARNING]
> **This is not a medical device and must not be used for diagnosis.**
> It is a research and teaching model. It gets about 3 in 10 images wrong, it was never tested in a clinic, and it can be confidently wrong. If you are worried about your skin, see a doctor or dermatologist.

---

## Contents

- [In plain words](#in-plain-words) (for everyone)
- [Quick start](#quick-start) (for developers)
- [The 23 categories](#the-23-categories)
- [Results](#results)
- [Model and training details](#model-and-training-details) (for researchers)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Repository files](#repository-files)
- [Citation and license](#citation-and-license)

---

## In plain words

**What it does:** you give it a photo of a skin problem, and it guesses which of 23 groups the condition belongs to, for example *"Eczema"*, *"Acne and Rosacea"*, or *"Melanoma, Nevi and Moles"*. It also says how sure it is about each guess.

**How good is it?** On 2,334 photos it had never trained on, its top guess was right **71% of the time**. That is useful for a student project or a demo. It is **not good enough** to make medical decisions.

**What it can't do:**
- Name a specific disease. Each category is a broad family of conditions.
- Say whether something is dangerous, or tell you what to do about it.
- Handle photos that don't look like the training photos, such as blurry or dim phone pictures, unusual angles, or skin tones that were rare in the training data.
- Say "this is healthy skin". Every image gets put into one of the 23 disease groups.

---

## Quick start

### 1. Get the model

The weights file (`model.safetensors`, ~343 MB) is stored with **Git LFS**. Install Git LFS before you clone, or you will only get a small text pointer instead of the real file.

```bash
git lfs install
git clone https://github.com/Muzammillcoste/dermnet-deep-learning-benchmark.git
cd dermnet-deep-learning-benchmark
```

### 2. Install dependencies

```bash
pip install "transformers>=4.38" torch pillow
```

### 3. Classify an image

**Easiest way (pipeline):**

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="./")  # path to this folder
results = classifier("my_skin_photo.jpg", top_k=5)

for r in results:
    print(f"{r['score']:.1%}  {r['label']}")
```

Example output format (the numbers are illustrative):

```
62.4%  Eczema Photos
14.1%  Atopic Dermatitis Photos
 7.9%  Psoriasis pictures Lichen Planus and related diseases
 ...
```

**Lower-level way (for batching or your own code):**

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_dir = "./"
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForImageClassification.from_pretrained(model_dir).eval()

image = Image.open("my_skin_photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)[0]

top5 = probs.topk(5)
for score, idx in zip(top5.values, top5.indices):
    print(f"{score:.1%}  {model.config.id2label[idx.item()]}")
```

**Tips**
- Show the **top 3 to 5** predictions, not only the first one. Many categories look alike.
- Treat the scores as a ranking. They are not calibrated probabilities.
- Runs fine on CPU, about 86M parameters per forward pass. A GPU helps for large batches.

---

## The 23 categories

| ID | Category | ID | Category |
|---:|---|---:|---|
| 0 | Acne and Rosacea | 12 | Nail Fungus and other Nail Disease |
| 1 | Actinic Keratosis, Basal Cell Carcinoma and other Malignant Lesions | 13 | Poison Ivy and other Contact Dermatitis |
| 2 | Atopic Dermatitis | 14 | Psoriasis, Lichen Planus and related diseases |
| 3 | Bullous Disease | 15 | Scabies, Lyme Disease and other Infestations and Bites |
| 4 | Cellulitis, Impetigo and other Bacterial Infections | 16 | Seborrheic Keratoses and other Benign Tumors |
| 5 | Eczema | 17 | Systemic Disease |
| 6 | Exanthems and Drug Eruptions | 18 | Tinea, Ringworm, Candidiasis and other Fungal Infections |
| 7 | Hair Loss, Alopecia and other Hair Diseases | 19 | Urticaria (Hives) |
| 8 | Herpes, HPV and other STDs | 20 | Vascular Tumors |
| 9 | Light Diseases and Disorders of Pigmentation | 21 | Vasculitis |
| 10 | Lupus and other Connective Tissue diseases | 22 | Warts, Molluscum and other Viral Infections |
| 11 | Melanoma Skin Cancer, Nevi and Moles | | |

The model returns the exact strings stored in [config.json](config.json) under `id2label`, for example `"Eczema Photos"` or `"Urticaria Hives"`. The table above cleans those names up for reading.

---

## Results

### Headline numbers

| Metric | Value |
|---|---|
| Validation accuracy (top-1) | **0.7099** (1,657 / 2,334 correct) |
| Validation loss (cross-entropy) | 1.1935 |
| Chance level (uniform, 23 classes) | 0.0435 |
| Final training loss (mean over run) | 0.532 |
| Best checkpoint | step 7,300 (epoch 8.83), selected by **lowest validation loss** |

The highest accuracy seen during training was 0.7112 at step 8,100. The saved model is the step-7,300 checkpoint because checkpoints were ranked by validation loss, not accuracy.

### Learning curve (end of each epoch)

| Epoch | Mean train loss | Val loss | Val accuracy |
|:---:|:---:|:---:|:---:|
| 1 | 1.099 | 1.446 | 0.574 |
| 2 | 0.816 | 1.451 | 0.589 |
| 3 | 0.608 | 1.472 | 0.601 |
| 4 | 0.518 | 1.402 | 0.626 |
| 5 | 0.605 | 1.242 | 0.665 |
| 6 | 0.485 | 1.304 | 0.658 |
| 7 | 0.385 | 1.284 | 0.679 |
| 8 | 0.325 | 1.265 | 0.696 |
| 9 | 0.263 | 1.203 | 0.704 |
| 10 | 0.220 | 1.200 | 0.711 |

Training loss drops much faster than validation loss, ending around 0.2 versus 1.2, which shows clear **overfitting**. Validation accuracy was still climbing slowly at epoch 10.

<details>
<summary>Full evaluation log (every 100 steps)</summary>

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.2171        | 0.12  | 100  | 1.5449          | 0.5390   |
| 1.1934        | 0.24  | 200  | 1.5524          | 0.5330   |
| 1.0425        | 0.36  | 300  | 1.4836          | 0.5570   |
| 1.2195        | 0.48  | 400  | 1.5463          | 0.5428   |
| 1.2398        | 0.6   | 500  | 1.5507          | 0.5548   |
| 1.0992        | 0.73  | 600  | 1.4974          | 0.5578   |
| 0.9141        | 0.85  | 700  | 1.4526          | 0.5801   |
| 0.9695        | 0.97  | 800  | 1.4464          | 0.5741   |
| 0.8629        | 1.09  | 900  | 1.5265          | 0.5677   |
| 0.7845        | 1.21  | 1000 | 1.5754          | 0.5583   |
| 0.7901        | 1.33  | 1100 | 1.5343          | 0.5690   |
| 0.8336        | 1.45  | 1200 | 1.4265          | 0.5891   |
| 0.7639        | 1.57  | 1300 | 1.5037          | 0.5750   |
| 0.8555        | 1.69  | 1400 | 1.4346          | 0.6011   |
| 0.8874        | 1.81  | 1500 | 1.3850          | 0.6003   |
| 0.7824        | 1.93  | 1600 | 1.4507          | 0.5891   |
| 0.6257        | 2.06  | 1700 | 1.4597          | 0.5925   |
| 0.6028        | 2.18  | 1800 | 1.4626          | 0.6054   |
| 0.6019        | 2.3   | 1900 | 1.5333          | 0.5664   |
| 0.5468        | 2.42  | 2000 | 1.4553          | 0.6007   |
| 0.5237        | 2.54  | 2100 | 1.4363          | 0.6015   |
| 0.6603        | 2.66  | 2200 | 1.4913          | 0.5750   |
| 0.5703        | 2.78  | 2300 | 1.4628          | 0.6071   |
| 0.4992        | 2.9   | 2400 | 1.4719          | 0.6011   |
| 0.4853        | 3.02  | 2500 | 1.4663          | 0.5887   |
| 0.3463        | 3.14  | 2600 | 1.5019          | 0.5947   |
| 0.4537        | 3.26  | 2700 | 1.5044          | 0.6037   |
| 0.4989        | 3.39  | 2800 | 1.4753          | 0.6135   |
| 0.4843        | 3.51  | 2900 | 1.4336          | 0.6221   |
| 0.4864        | 3.63  | 3000 | 1.4612          | 0.6161   |
| 0.3936        | 3.75  | 3100 | 1.4786          | 0.6217   |
| 0.6484        | 3.87  | 3200 | 1.4947          | 0.6148   |
| 0.766         | 3.99  | 3300 | 1.4022          | 0.6255   |
| 0.5875        | 4.11  | 3400 | 1.3863          | 0.6315   |
| 0.6366        | 4.23  | 3500 | 1.4059          | 0.6418   |
| 0.4798        | 4.35  | 3600 | 1.3654          | 0.6362   |
| 0.5828        | 4.47  | 3700 | 1.4061          | 0.6260   |
| 0.5378        | 4.59  | 3800 | 1.3399          | 0.6560   |
| 0.5519        | 4.72  | 3900 | 1.3586          | 0.6350   |
| 0.6189        | 4.84  | 4000 | 1.3274          | 0.6465   |
| 0.6252        | 4.96  | 4100 | 1.2417          | 0.6650   |
| 0.4329        | 5.08  | 4200 | 1.3092          | 0.6628   |
| 0.5504        | 5.2   | 4300 | 1.3120          | 0.6547   |
| 0.5053        | 5.32  | 4400 | 1.3241          | 0.6482   |
| 0.4077        | 5.44  | 4500 | 1.2671          | 0.6684   |
| 0.5016        | 5.56  | 4600 | 1.3034          | 0.6641   |
| 0.4671        | 5.68  | 4700 | 1.3233          | 0.6525   |
| 0.5919        | 5.8   | 4800 | 1.3478          | 0.6607   |
| 0.5295        | 5.93  | 4900 | 1.3041          | 0.6577   |
| 0.3118        | 6.05  | 5000 | 1.2377          | 0.6731   |
| 0.3774        | 6.17  | 5100 | 1.2894          | 0.6607   |
| 0.405         | 6.29  | 5200 | 1.2821          | 0.6735   |
| 0.3187        | 6.41  | 5300 | 1.2697          | 0.6727   |
| 0.4335        | 6.53  | 5400 | 1.3005          | 0.6645   |
| 0.3935        | 6.65  | 5500 | 1.2890          | 0.6701   |
| 0.5328        | 6.77  | 5600 | 1.3079          | 0.6752   |
| 0.3797        | 6.89  | 5700 | 1.2841          | 0.6787   |
| 0.353         | 7.01  | 5800 | 1.2331          | 0.6808   |
| 0.3576        | 7.13  | 5900 | 1.2487          | 0.6787   |
| 0.3157        | 7.26  | 6000 | 1.2325          | 0.6834   |
| 0.3551        | 7.38  | 6100 | 1.2531          | 0.6817   |
| 0.261         | 7.5   | 6200 | 1.2243          | 0.6979   |
| 0.3384        | 7.62  | 6300 | 1.2787          | 0.6821   |
| 0.1776        | 7.74  | 6400 | 1.2401          | 0.7001   |
| 0.3227        | 7.86  | 6500 | 1.2233          | 0.6941   |
| 0.1673        | 7.98  | 6600 | 1.2653          | 0.6958   |
| 0.1985        | 8.1   | 6700 | 1.2421          | 0.6911   |
| 0.2384        | 8.22  | 6800 | 1.2494          | 0.6915   |
| 0.3055        | 8.34  | 6900 | 1.2675          | 0.6937   |
| 0.3417        | 8.46  | 7000 | 1.2517          | 0.6967   |
| 0.3827        | 8.59  | 7100 | 1.2827          | 0.6911   |
| 0.2781        | 8.71  | 7200 | 1.2234          | 0.6979   |
| 0.3134        | 8.83  | 7300 | 1.1935          | 0.7099   |
| 0.2248        | 8.95  | 7400 | 1.2028          | 0.7044   |
| 0.2491        | 9.07  | 7500 | 1.2043          | 0.7108   |
| 0.2153        | 9.19  | 7600 | 1.2054          | 0.7057   |
| 0.2619        | 9.31  | 7700 | 1.2102          | 0.7035   |
| 0.2425        | 9.43  | 7800 | 1.2161          | 0.7078   |
| 0.2068        | 9.55  | 7900 | 1.2068          | 0.7069   |
| 0.222         | 9.67  | 8000 | 1.2035          | 0.7091   |
| 0.0899        | 9.79  | 8100 | 1.2022          | 0.7112   |
| 0.2154        | 9.92  | 8200 | 1.1999          | 0.7108   |

</details>

You can view the full curves with TensorBoard:

```bash
pip install tensorboard
tensorboard --logdir runs/
```

---

## Model and training details

### Architecture

| | |
|---|---|
| Base model | [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k), pre-trained on ImageNet-21k |
| Type | `ViTForImageClassification` (12 layers, 12 heads, hidden size 768, about 86M parameters) |
| Head | Linear classifier on the `[CLS]` token, 23 outputs, softmax / cross-entropy |
| Input | 224 × 224 RGB, 16 × 16 patches (196 tokens + `[CLS]`) |
| Preprocessing | Resize to 224 × 224 (bilinear), scale to [0, 1], normalize with mean = std = 0.5 per channel |
| Precision | Weights stored as float32 (`model.safetensors`) |

### Data

- **Source:** the DermNet image collection, in the 23-category folder layout used by the public Kaggle "Dermnet" dataset. The label names match those folder names.
- **Split:** the logs show **2,334 validation images** and about **13.2k training images**, roughly an 85/15 split. The split script and seed are not in this repo.
- **Note on evaluation:** the validation set was also used to pick the best checkpoint. The reported 71.0% is therefore slightly optimistic. Scores on a separate held-out test set have not been reported.

### Hyperparameters

| Setting | Value |
|---|---|
| Learning rate | 2e-4, linear decay, no warmup |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8) |
| Batch size | 16 train / 8 eval |
| Epochs | 10 (8,270 steps) |
| Mixed precision | Native AMP (fp16) |
| Seed | 42 |
| Eval / save interval | every 100 steps |
| Model selection | lowest validation loss (`load_best_model_at_end`) |

### Compute

- About 1 h 44 min of training (6,220 s), around 21 images/s, on a single GPU.
- Evaluation takes about 33 s for 2,334 images, around 70 images/s.

### Reproducibility notes

- At step 10 the training loss was already about 1.23, far below the ≈3.1 (ln 23) you would expect from a freshly initialized head. This, together with several run folders under `runs/`, suggests the run **continued from an earlier fine-tuning session**. The hyperparameters above describe the final session only.
- Framework versions: Transformers 4.38.1, PyTorch 2.1.2, Datasets 2.1.0, Tokenizers 0.15.2.

---

## Limitations and responsible use

**Intended uses**
- Teaching and coursework on transfer learning and vision transformers
- A baseline for comparing other dermatology classifiers
- Research prototypes, always with a human expert in the loop

**Out-of-scope uses**
- Diagnosing, triaging or screening real patients
- Any app that tells the public what condition they have
- Deciding whether a lesion is cancer

**Known limitations**
- **About 29% error rate.** Some errors may be dangerous, for example a malignant lesion classified as benign.
- **Overlapping categories.** Several classes are clinically close, such as Eczema vs. Atopic Dermatitis, or Psoriasis vs. Lichen Planus. Expect confusion between them.
- **Dataset bias.** DermNet is an atlas of mostly clinical photos. Lighter skin tones are believed to be over-represented, and per-skin-tone performance has not been measured. Results on darker skin, phone photos or dermoscopy images may be much worse.
- **Closed set.** No "normal skin" or "unknown" class exists. Every input is forced into one of the 23 categories, including photos that are not skin at all.
- **Uncalibrated confidence.** A high score does not mean the prediction is reliable.
- **Missing analysis.** Per-class metrics, a confusion matrix and a test-set evaluation have not been computed yet. These are good next steps.

---

## Repository files

| File | What it is |
|---|---|
| `model.safetensors` | Trained weights (Git LFS, ~343 MB) |
| `config.json` | Model architecture and the `id2label` / `label2id` maps |
| `preprocessor_config.json` | Image resize and normalization settings |
| `training_args.bin` | Pickled `TrainingArguments` from the Hugging Face `Trainer` |
| `trainer_state.json` | Full training and evaluation log history |
| `all_results.json`, `train_results.json`, `eval_results.json` | Final summary metrics |
| `runs/` | TensorBoard event files |

---

## Citation and license

Released under the **Apache 2.0** license, the same license as the base ViT model. The DermNet images belong to their original copyright holders. Check DermNet's terms before you reuse the images.

If you use this model, please cite:

```bibtex
@misc{finetuned_dermnet,
  title  = {finetuned-dermnet: ViT-Base/16 fine-tuned on DermNet (23 classes)},
  author = {Muzammil},
  year   = {2024},
  url    = {https://github.com/Muzammillcoste/dermnet-deep-learning-benchmark}
}

@inproceedings{dosovitskiy2021vit,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and others},
  booktitle = {ICLR},
  year      = {2021}
}
```

**Acknowledgments:** Google Research for ViT, Hugging Face for `transformers`, and DermNet for the image collection.
