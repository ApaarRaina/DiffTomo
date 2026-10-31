# Contributing to DiffTomo

Thank you for your interest in DiffTomo. Contributions from everyone are welcome, whether that is a bug fix, a new method, documentation or tests.

## Requesting a feature, reporting a bug or suggesting a change

Please [open an issue](https://github.com/ApaarRaina/DiffTomo/issues) describing what you would like to see. For bugs, include your environment (OS, Python, PyTorch and CUDA versions) and the steps needed to reproduce the problem.

## Contributing code

1. Fork the repository and create a branch for your change.
2. Follow the conventions already used in the codebase: file layout, naming, docstrings and code style. If you are unsure, look at how similar code is written in the repository and match it.
3. Install `pre-commit` and enable it so your changes are formatted the same way as the rest of the repository:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   The hooks run automatically on each commit. To check everything at once, run `pre-commit run --all-files`.
4. Open a pull request and describe what it changes and why. If it addresses an issue, reference it.

## Using AI tools

AI-assisted contributions are fine, but the same rules apply to them. Please install `pre-commit` and let it format your changes before you commit, so that AI-generated code matches the formatting and conventions of the repository. You are responsible for understanding, testing and reviewing anything you submit.