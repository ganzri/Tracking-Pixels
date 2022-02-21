# Script to extract information from images stored in leveldb

import plyvel
from PIL import Image
import io
import PIL
import warnings
from typing import Tuple, List, Dict, Any, Optional, Set


# access leveldb
db = plyvel.DB('/mnt/a9fcadb8-dc39-4b73-a3f9-3477390868ec/Crawl_Data/content.ldb', create_if_missing=False)

#get content_hash from http_responses and query db with it
#img=db.get(b'content_hash')
warnings.simplefilter('error', Image.DecompressionBombWarning)
warnings.simplefilter('error', UserWarning)


# or iterate over key, value pairs
count = 0
format_set:  Set = set()
mode_set: Set = set()
for key, value in db:
    try:
        image = Image.open(io.BytesIO(value))
        format_set.add(image.format)
        mode_set.add(image.mode)
        count += 1
        print(count)
    except (PIL.UnidentifiedImageError, OSError, ValueError, UserWarning, Image.DecompressionBombError, Image.DecompressionBombWarning):
            continue

print(format_set)
print(mode_set)
db.close()
