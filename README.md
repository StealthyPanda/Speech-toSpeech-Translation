# Speech to Speech Translation

## Setup & Running
The entire pipeline is contained within `Archs.py`. An example training pipeline is shown in `Training.ipynb`. The corresponding token banks are also included for English and Swahili in 128 and 64 vocab sizes, in `.token` files, which can just be loaded using `pickle`.

The models can be loaded using `talos`, which can be installed using:


For python > 3.10:
```bash
pip install git+https://github.com/StealthyPanda/taloslib
```

For python <= 3.10 or on kaggle or colab:

```bash
pip install git+https://github.com/StealthyPanda/taloslib@kaggle
```