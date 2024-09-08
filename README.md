# Accessing Output Privacy of Different Fine-tuning methods on LLMs

Author: Olivia Ma

Install the required library:
```
!pip3 install -U tensorflow==2.14.0 tensorflow_empirical_privacy==0.1.0 opacus==1.5.2 transformers==4.35.2 accelerate==0.24.1
```

Install dp-transformers 1.0.1:
1. Clone dp-transformers from https://github.com/microsoft/dp-transformers
2. install the latest version (1.0.1) directly in the repo
```
cd dp-transformers
pip3 install .
```

Run the *_main.py files for training and MIA attack, defaulting to DistilBERT with IMDB dataset.
