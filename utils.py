from functools import lru_cache

import cv2

try:
    import tkinter as tk
except Exception:
    tk = None


@lru_cache(maxsize=1)
def get_screen_size():
    """Return the primary screen size in pixels."""
    try:
        if tk is None:
            raise RuntimeError("tkinter is unavailable")
        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        # Reasonable fallback when Tk is unavailable in the current environment.
        return 1920, 1080


def resize_to_fit(image, max_w=None, max_h=None, padding=80, return_scale=False):
    """
    Resize an image so it fits on screen without changing aspect ratio.

    The default behavior avoids upscaling. When `return_scale` is True, the
    resized image and the applied scale factor are returned.
    """
    if image is None:
        raise ValueError("Image cannot be None.")

    img_h, img_w = image.shape[:2]
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image must have positive width and height.")

    screen_w, screen_h = get_screen_size()
    fit_w = max_w if max_w is not None else max(1, screen_w - padding)
    fit_h = max_h if max_h is not None else max(1, screen_h - padding)

    scale = min(fit_w / img_w, fit_h / img_h, 1.0)

    if scale == 1.0:
        resized = image.copy()
    else:
        out_w = max(1, int(round(img_w * scale)))
        out_h = max(1, int(round(img_h * scale)))
        resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    if return_scale:
        return resized, scale
    return resized


def center_window(win_name, window_w, window_h):
    """Center an OpenCV window on the primary screen."""
    screen_w, screen_h = get_screen_size()
    x = max(0, (screen_w - window_w) // 2)
    y = max(0, (screen_h - window_h) // 2)
    cv2.moveWindow(win_name, x, y)
