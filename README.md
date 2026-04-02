# MATCHa

MATCHa is a semantic schema matching algorithm designed to combine different matchers to get the best results.

## Installation

In order to install MATCHa, clone the repository and open a terminal in an IDE of your choice, then run the following commands:

- `python -m venv .venv`


- On Windows: `.venv\Scripts\activate`
- On Linux/Mac:`source .venv/bin/activate`


- `python -m pip install -r requirements.txt`
- `python -m pip install -e .`

You should now be able to run the project. A simple demo can be found in `src/run.py`.
If you are running into import errors or unresolved references make sure the virtual environment is activated, and check whether the correct interpreter is selected in your IDE. The path should look like this:

- `.venv/Scripts/python.exe`

or like this:

- `.venv/bin/python`

Run run.py for a small demonstration of MATCHa. Please note that no embeddings are generated in this demo, as access to Huggingface and the EmbeddingGemma model cannot be guaranteed.