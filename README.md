# Diffusion Transformer for MNIST number generation

Trained Diffusion Transformer for generating MNIST numbers with Flow Matching algorithm.

# Papers
- Diffusion Transformer: https://arxiv.org/pdf/2212.09748
- Flow Matching algorithm: https://arxiv.org/pdf/2210.02747

# How to run

```bash

git clone https://github.com/morikonon/mnist-generator.git

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```