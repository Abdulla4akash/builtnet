"""Read inputs shared by the paper generators without building any network."""
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
CARDS = ('nl-2018', 'nl-series', 'dk', 'se')
LAYERS = ('household', 'neighbour', 'school', 'work', 'kin')


def json_read(path):
    return json.loads(path.read_text())


def card_files():
    return {p.stem: p for p in sorted((REPO / 'src' / 'builtnet' / 'cards').glob('*.toml'))}
