import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_1 = loadimage.load_image(image="ComfyUI_temp_jbijt_00006_.png")

        loadimage_4 = loadimage.load_image(image="ComfyUI_temp_jbijt_00006_.png")

        sammodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "SAMModelLoader (segment anything)"
        ]()
        sammodelloader_segment_anything_5 = sammodelloader_segment_anything.main(
            model_name="sam_vit_b (375MB)"
        )

        groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoModelLoader (segment anything)"
        ]()
        groundingdinomodelloader_segment_anything_7 = (
            groundingdinomodelloader_segment_anything.main(
                model_name="GroundingDINO_SwinB (938MB)"
            )
        )

        groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoSAMSegment (segment anything)"
        ]()
        catvtonwrapper = NODE_CLASS_MAPPINGS["CatVTONWrapper"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            groundingdinosamsegment_segment_anything_6 = (
                groundingdinosamsegment_segment_anything.main(
                    prompt="",
                    threshold=0.3,
                    sam_model=get_value_at_index(sammodelloader_segment_anything_5, 0),
                    grounding_dino_model=get_value_at_index(
                        groundingdinomodelloader_segment_anything_7, 0
                    ),
                    image=get_value_at_index(loadimage_1, 0),
                )
            )

            catvtonwrapper_2 = catvtonwrapper.catvton(
                mask_grow=25,
                mixed_precision="fp16",
                seed=random.randint(1, 2**64),
                steps=40,
                cfg=2.5,
                image=get_value_at_index(loadimage_1, 0),
                mask=get_value_at_index(groundingdinosamsegment_segment_anything_6, 1),
                refer_image=get_value_at_index(loadimage_4, 0),
            )

            saveimage_8 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(catvtonwrapper_2, 0),
            )


if __name__ == "__main__":
    main()
