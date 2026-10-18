# DiffTomo

A PyTorch library for **differentiable tomographic and medical image reconstruction**, bringing together classical and deep learning methods in one place.

DiffTomo is designed to cover multiple imaging modalities, including CT, MRI and PET, with differentiable forward and backward operators that can sit directly inside neural network pipelines. At the moment the library contains CT functions only. MRI and PET reconstruction are planned.

## Features

### Currently available (CT)

- **Differentiable projectors** implemented as CUDA kernels with autograd support
  - Parallel-beam: forward projection, backprojection and filtered backprojection (FBP)
  - Fan-beam: forward projection, backprojection and filtered backprojection (FBP)
- **FBP filters:** `ram-lak`, `shepp-logan`, `cosine`, `hamming`, `hann`
- **Iterative reconstruction:** SIRT, TV regularization, TGV regularization
- **Data:** Shepp-Logan phantom and sample datasets (carved cheese, walnut)
- **Pretrained models:** loading support for a small set of models

### Planned

- MRI reconstruction
- PET reconstruction
- More classical and deep learning methods across modalities

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- An NVIDIA GPU and a CUDA toolkit (`nvcc`) matching your PyTorch build
- `ninja` (the CUDA extension is compiled just in time)

## Documentation

For guidelines and how to use the library, see the [docs](docs/).

## Installation

```bash
git clone https://github.com/yourusername/difftomo.git
cd difftomo
pip install -e .
```

## Repository layout

```
difftomo/         # library source
  CT/             # projectors and iterative methods
  data/           # phantoms and datasets
docs/             # documentation and examples
test/             # tests
```

## Roadmap

- [ ] ART and SART
- [ ] Helsinki CT dataset and more pretrained models
- [ ] A learned, transformer attention-based iterative filter that replaces the fixed FBP filter, to study how the filter affects reconstruction quality
- [ ] MRI reconstruction functions
- [ ] PET reconstruction functions