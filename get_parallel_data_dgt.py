import requests
import json
import os
import subprocess
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Download DGT Translation Memories via data.europa.eu API
res = requests.get("https://data.europa.eu/api/hub/search/datasets/dgt-translation-memory")
res = json.loads(res.text)

data = [
    {k: v for k, v in d.items() if k in ["download_url", "status", "issued"]}
    for d in res["result"]["distributions"]
]
df = pd.DataFrame(data)

df = pd.concat([df[["download_url", "issued"]], pd.json_normalize(df["status"])], axis=1)
df["download_url"] = df["download_url"].str[0]

os.makedirs("DGT-TM", exist_ok=True)

for url in df["download_url"]:
    filename = url.split("/")[-1]
    with open(os.path.join("DGT-TM", filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)


zip_files = [file for file in os.listdir("DGT-TM") if file.endswith(".zip")]
os.makedirs("DGT-TM/data", exist_ok=True)

for zip_file in zip_files:
    with zipfile.ZipFile(os.path.join("DGT-TM", zip_file), "r") as zip_ref:
        zip_ref.extractall("DGT-TM/data")


subprocess.check_call("cwm --rdf test.rdf --ntriples > test.nt", shell=True)

tree = ET.parse("DGT-TM/dgttm.tmx")

for tu in tqdm(tree.findall(".//body/tu"), total=len(tree.findall(".//body/tu"))):
    if len(tu.findall(".//tuv")) < 2:
        print("Unpaired translation. Ignoring...")
    else:
        # Get language attributes
        srclang = tu.find(".//tuv").attrib["{http://www.w3.org/XML/1998/namespace}lang"]
        targetlang = tu.find(".//tuv[2]").attrib["{http://www.w3.org/XML/1998/namespace}lang"]
        # Get source sentence
        srcsentence = tu.find(".//tuv/seg").text
        # Get target sentence
        targetsentence = tu.find(".//tuv[2]/seg").text

    # Write srcsentence and targetsentence to tsv file append mode, using tab as delimiter
    with open("parallel-sentences/DGT-TM-en-sv.tsv", "a") as f:
        _ = f.write(srcsentence + "\t" + targetsentence + "\n")
