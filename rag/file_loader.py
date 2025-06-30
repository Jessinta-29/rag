def load_file(file):
    import tempfile
    import os
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

    ext = file.name.split(".")[-1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), file.name)

    file.seek(0)
    with open(temp_path, "wb") as f:
        f.write(file.read())

    if os.path.getsize(temp_path) == 0:
        raise ValueError("Uploaded file is empty or failed to save correctly.")

    if ext == "pdf":
        loader = PyPDFLoader(temp_path)
    elif ext == "csv":
        loader = CSVLoader(temp_path)
    elif ext == "txt":
        loader = TextLoader(temp_path)
    else:
        return None

    return loader.load()
