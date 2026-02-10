import pathlib
from PIL import Image

PROC_ROOT = pathlib.Path("data/Processed/yolo_malaria/images/train")
RAW_ROOT = pathlib.Path("data/Raw/ThickBloodSmears_150/TF127_W14_62")

IMG_NAMES = [
    "20170607_142842.jpg",
    "20170612_143853.jpg",
    "20170613_160430.jpg",
    "20170707_150317.jpg",
    "20170708_101907.jpg",
    "20170831_145253.jpg",
    "20170831_145423.jpg",
]

ORIENTATION_TAG = 274  # EXIF orientation

def describe(path: pathlib.Path) -> str:
    if not path.exists():
        return "missing"
    img = Image.open(path)
    orient = img.getexif().get(ORIENTATION_TAG)
    return f"size={img.size}, orientation_tag={orient}"


for name in IMG_NAMES:
    proc_desc = describe(PROC_ROOT / name)
    raw_desc = describe(RAW_ROOT / name)
    print(f"{name}: processed: {proc_desc} | raw: {raw_desc}")
