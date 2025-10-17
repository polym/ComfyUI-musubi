from .modules.nodes import MusubiUNETLoader, MusubiUNETJitLoader

NODE_CLASS_MAPPINGS = {
    "MusubiUNETLoader": MusubiUNETLoader,
    "MusubiUNETJitLoader": MusubiUNETJitLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusubiUNETLoader": "Musubi UNET Loader",
    "MusubiUNETJitLoader": "Musubi UNET JustInTime Loader",
}
