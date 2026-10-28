"""Explicit user-selected GOOSE-Ex download only. No image is bundled here."""
import argparse,urllib.request
from pathlib import Path
from urllib.parse import urlparse
def main():
    p=argparse.ArgumentParser(description='Obtain an official dataset URL after reviewing https://goose-dataset.de/docs/setup/ and the applicable terms. No automatic license acceptance.');p.add_argument('--official-url',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    u=urlparse(a.official_url)
    if u.scheme!='https':raise SystemExit('HTTPS required.')
    if a.output.exists():raise SystemExit('Refusing to overwrite an existing file.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(a.official_url) as src,a.output.open('xb') as dst:
        while block:=src.read(1<<20):dst.write(block)
if __name__=='__main__':main()
