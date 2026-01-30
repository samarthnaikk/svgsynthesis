Here is a **single, clean `README.md`** you can use directly for **samarthnaikk**.
It is concise, properly attributed, and clearly states the macOS-specific difference.

---

# Handwriting Synthesis (macOS)

> **Maintainer:** samarthnaikk
> **Derived from:** [https://github.com/otuva/handwriting-synthesis](https://github.com/otuva/handwriting-synthesis)

This repository is a **macOS-compatible adaptation** of the original *Handwriting Synthesis* project by **otuva**, which implements the experiments from the paper:

**Generating Sequences with Recurrent Neural Networks**
Alex Graves — [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850)

The core model, architecture, and results remain unchanged.
This version focuses on **smooth setup and execution on macOS**, including Apple Silicon systems.

---

## Features

* TensorFlow v2 implementation of handwriting synthesis
* Pretrained model included
* SVG handwriting generation
* Verified to work on macOS (Intel & Apple Silicon)

---

## Installation (macOS)

### Requirements

* Python 3.8–3.10
* macOS 11+ (Big Sur or later)

### Setup

```bash
git clone https://github.com/samarthnaikk/handwriting-synthesis.git
cd handwriting-synthesis

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow will install the macOS-optimized build automatically.

---

## Running the Demo

```bash
python main.py
```

This generates handwriting samples as SVG files using a pretrained model.

---

## Usage Example

```python
from handwriting_synthesis.hand import Hand

lines = [
    "Father time, I'm running late",
    "I'm winding down, I'm growing tired",
    "Seconds drift into the night",
    "The clock just ticks till my time expires",
]

biases = [0.75 for _ in lines]
styles = [9 for _ in lines]

hand = Hand()
hand.write(
    filename='output.svg',
    lines=lines,
    biases=biases,
    styles=styles
)
```

---

## Training

A pretrained model is included.
To train from scratch, follow the instructions in:

```
model/README.md
```

---

## Attribution

This project is derived from **otuva/handwriting-synthesis**.
All original research and implementation credit belongs to the original author.

---

## License

Same license as the upstream project.
Refer to the original repository for details.

---
