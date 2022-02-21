import json
from typing import List, Dict, Any, Tuple

def main() -> None:
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    out_pixels: Dict[str, Dict[str, Any]] = dict()

    with open('10_12_2021_13_40.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")

    #get all functional pixels
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if cat == 1:
            out_pixels[key] = sample
            url = sample["url"]
            g ="google.com"
            f ="facebook.com"
            #"truly" functional
            aw = "forms.aweber.com"
            ny = "nyhedsbrev.bog.dk"
            co = "colombo.pt"
            #ab = "a.b0e8.com" on tracking server list, should probably not be in functional, but debatable
            rd = "da.rodekors.dk"

            if g in url:
                continue  #google ones get reclassified later by the filter, as they can be 2 or 3
            elif f in url:
                sample["label"] = 3
            elif aw in url or ny in url or co in url or rd in url:
                continue 
            else: #everyone else goes to cat 2
                sample["label"] = 2
            out_pixels[key] = sample
                
        else: #keep all other pixels as they are
            out_pixels[key] = sample

    print(f"Nr of samples returned: {len(out_pixels)}")
    with open("all_cat_only_functional_filtered.json", "w") as outf:
        json.dump(out_pixels, outf)

if __name__ == "__main__":
    exit(main())
