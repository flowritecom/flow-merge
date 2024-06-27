def has_adapter_files(file_list):
    return any(
        file.startswith("adapter_")
        and (file.endswith(".bin") or file.endswith(".safetensors"))
        for file in file_list
    )