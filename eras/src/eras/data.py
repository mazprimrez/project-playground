"""The Eras Tour outfit + surprise song data, from the fan-made "Eras Tour" Google Sheet, as tidy tables.

Two tabs are used (downloaded as CSV with `download`):

- ERAS TOUR 2023-2024: one row per show, one column per outfit version (header rows: era, version, name, garment);
  a cell is TRUE when that version was worn.
- SURPRISE SONGS: one column per show; the guitar songs (rows 7-9) and piano songs (rows 10-13) of the surprise
  set (in 2024 often mashups, so up to 3 songs per instrument), then the song catalogue by album with a True / False
  "played" matrix.

Game categories follow the Swift Alert Mastermind game: 15 picks per show, 4 points each (60 in total); the surprise
songs are one guess per instrument, 2 points each, correct if the song was played on that instrument that night.
The Lover guitar is not tracked in the sheet, so 14 categories (56 points) can be predicted.
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SHEET_ID = "1caBoaDi1id6k4s1sP1bTbo846inOXzN13p0C5OrFY2E"
TABS = {"outfits.csv": 393964062, "songs.csv": 1928613815}
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"

# game category -> (era, garment label, part) in the outfit tab; part > 0 = a block that restarts at V1 under the same label
GAME = {
    "Lover bodysuit": ("LOVER", "Bodysuit", 0),
    "The Man jacket": ("LOVER", "Blazer Jacket", 0),
    "Lover guitar": None,                                            # not in the sheet
    "Fearless dress": ("FEARLESS", None, 0),
    "Red shirt": ("RED", None, 0),
    "Speak Now dress": ("SPEAK NOW", None, 0),
    "Reputation outfit": ("REP", None, 0),
    "Folkmore dress": ("FOLK(L/M)ORE", None, 0),
    "1989 set": ("1989", None, 0),
    "TTPD outfit": "combo",                                          # dress + two piece / jacket
    "Surprise song dress": ("SURPRISE", None, 0),
    "Surprise songs": "songs",
    "Midnights shirt": ("MIDNIGHTS", "Sequin Shirt Dress", 0),
    "Midnights bodysuit": ("MIDNIGHTS", "Bodysuit", 0),
    "Karma jacket": ("MIDNIGHTS", '"Karma" Jacket', 0),
}
TTPD_DRESS = ("TTPD", "Jacket + Dress / Gloves", 0)
TTPD_SET = ("TTPD", '"I Can Do It With A Broken Heart"', 0)
OUTFIT_CATEGORIES = [c for c, v in GAME.items() if v not in (None, "songs")]
TTPD_PARTS = ("TTPD dress", "TTPD two piece")      # predicted separately; the TTPD outfit pick is "dress + two piece"
POINTS = 4                        # per outfit category
SONG_POINTS = 2                   # per instrument

ALBUMS = ["DEBUT", "FEARLESS", "SPEAK NOW", "RED", "1989", "REPUTATION", "LOVER", "FOLKLORE", "EVERMORE", "MIDNIGHTS",
          "TTPD"]
ALIASES = {"thankuaimee": "thanK you aIMee", "thetorturedpoetsdepartment": "The Tortured Poets Dpt"}
# the sheet's catalogue abbreviates some titles; shown in full (cells using either form still match)
FULL_TITLES = {
    "IOMWIWY": "I'm Only Me When I'm With You", "TOSOTD": "The Other Side Of The Door",
    "TWAF": "Today Was A Fairytale", "WANEGBT": "We Are Never Ever Getting Back Together",
    "CBBH": "Come Back... Be Here", "IBYTAM": "I Bet You Think About Me", "OOTW": "Out Of The Woods",
    "AYHTDWS": "All You Had To Do Was Stay", "HYGTG": "How You Get The Girl", "YAIL": "You Are In Love",
    "LWYMMD": "Look What You Made Me Do", "KOMH": "King Of My Heart", "DWOHT": "Dancing With Our Hands Tied",
    "TIWWCHNT": "This Is Why We Can't Have Nice Things", "CIWYW": "Call It What You Want",
    "IFTYE": "I Forgot That You Existed", "ITHK": "I Think He Knows", "DBATC": "Death By A Thousand Cuts",
    "SYGB": "Soon You'll Get Better", "YNTCD": "You Need To Calm Down", "INTHAF": "It's Nice To Have A Friend",
    "AOTGYLB": "All Of The Girls You Loved Before", "YOYOK": "You're On Your Own, Kid",
    "BTTWS": "Bigger Than The Whole Sky", "WCS": "Would've, Could've, Should've",
}

LEGS = [("U.S. 2023", "2023-03-01"), ("Latin America 2023", "2023-08-20"), ("Asia & Australia 2024", "2024-01-01"),
        ("Europe 2024", "2024-04-01"), ("North America 2024", "2024-10-01")]


def download(data_dir, timeout: int = 60) -> None:
    """Fetch the two tabs as CSV into data_dir (the sheet must be shared as 'anyone with the link')."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, gid in TABS.items():
        with urllib.request.urlopen(f"{SHEET_URL}/export?format=csv&gid={gid}", timeout=timeout) as r:
            (data_dir / name).write_bytes(r.read())


def clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if isinstance(text, str) else ""


@dataclass
class Tour:
    """Everything a model needs, in show order (show = 0 .. n-1)."""
    shows: pd.DataFrame                          # show, date, city, night, leg
    outfits: dict = field(default_factory=dict)  # category -> [version label or None per show]
    guitar: list = field(default_factory=list)   # per show: set of songs played on guitar
    piano: list = field(default_factory=list)    # per show: set of songs played on piano
    catalogue: pd.DataFrame | None = None        # song, album, main_set (in the regular setlist: never a surprise)

    @property
    def n(self) -> int:
        return len(self.shows)


# ---------------------------------------------------------------- outfits

def parse_outfits(path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, header=None, dtype=str)
    era = raw.iloc[0].ffill()
    # column -> (era, garment, part): a new block starts at a garment label, a new era, or a version restarting at V1
    block, cur, part = {}, None, 0
    for i in range(2, raw.shape[1]):
        label = clean(raw.iloc[3, i]) or None
        new_era = i == 2 or era[i] != era[i - 1]
        if label or new_era:
            cur, part = label, 0
        elif clean(raw.iloc[1, i]) == "V1":
            part += 1
        block[i] = (era[i], cur, part)
    names = {i: clean(raw.iloc[2, i]) or clean(raw.iloc[1, i]) for i in block}

    body = raw.iloc[4:].reset_index(drop=True)
    body = body[body[0].notna() & body[0].str.match(r"\d+/\d+/\d{4}")].reset_index(drop=True)
    city_night = body[1].map(clean)
    shows = pd.DataFrame({
        "show": range(len(body)),
        "date": pd.to_datetime(body[0], format="%m/%d/%Y"),
        "city": city_night.str.replace(r"\s+\d+$", "", regex=True),
        "night": city_night.str.extract(r"(\d+)$")[0].fillna("1").astype(int),
    })
    shows["leg"] = pd.cut(shows.date, [pd.Timestamp(d) for _, d in LEGS] + [pd.Timestamp.max],
                          labels=[name for name, _ in LEGS], right=False).astype(str)

    def worn(key):
        cols = [i for i, b in block.items() if b == key]
        assert cols, f"no columns for {key}"
        return [next((names[i] for i in cols if body.iloc[t, i] == "TRUE"), None) for t in range(len(body))]

    outfits = {}
    for cat, key in GAME.items():
        if key == "combo":
            dress, two_piece = worn(TTPD_DRESS), worn(TTPD_SET)
            outfits[cat] = [f"{d} + {s}" if d and s else None for d, s in zip(dress, two_piece)]
            outfits[TTPD_PARTS[0]], outfits[TTPD_PARTS[1]] = dress, two_piece
        elif isinstance(key, tuple):
            outfits[cat] = worn(key)
    return shows, outfits


# ---------------------------------------------------------------- surprise songs

def norm(title: str) -> str:
    t = clean(title).lower().replace("&", "and").replace("’", "'")
    t = re.sub(r"\(.*?\)", "", t)                      # (Taylor's Version), (10 Minute Version) ...
    return re.sub(r"[^a-z0-9]", "", t)


def acronym(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", re.sub(r"\(.*?\)", "", clean(title)))
    return "".join(w[0] for w in words).lower()


def parse_songs(path, n_shows: int) -> tuple[list, list, pd.DataFrame]:
    raw = pd.read_csv(path, header=None, dtype=str)
    dates = raw.iloc[0, 1:]
    cols = [c for c in raw.columns[1:] if isinstance(dates[c], str) and re.match(r"\d+/\d+/\d{4}", dates[c])]
    assert len(cols) == n_shows, f"{len(cols)} show columns in the songs tab, {n_shows} shows in the outfit tab"

    # the catalogue: album headers, each followed by its songs; a song without any True/False is in the main setlist
    start = raw.index[raw[0].map(clean) == "DEBUT"][0]
    rows, album = [], None
    for r in range(start, len(raw)):
        label = clean(raw.iloc[r, 0])
        if not label:
            continue
        if label in ALBUMS:
            album = label
            continue
        flags = raw.iloc[r, cols].map(clean)
        rows.append({"song": label, "album": album, "main_set": not flags.isin(["True", "False", "TRUE", "FALSE"]).any()})
    catalogue = pd.DataFrame(rows).drop_duplicates("song").reset_index(drop=True)
    catalogue["song"] = catalogue.song.replace(FULL_TITLES)

    by_norm = {norm(s): s for s in catalogue.song}
    by_acr = {}
    for s in catalogue.song:
        by_acr.setdefault(acronym(s), []).append(s)

    def match(cell: str) -> str:
        """A song name as written in the surprise rows -> its catalogue title (unknown songs are kept as is)."""
        key = norm(cell)
        if key in by_norm:
            return by_norm[key]
        if key in ALIASES:
            return ALIASES[key]
        if acronym(cell) in by_norm and len(acronym(cell)) >= 3:   # catalogue abbreviated: "CIWYW" <- Call It What You Want
            return by_norm[acronym(cell)]
        if len(by_acr.get(key, [])) == 1 and len(key) >= 3:        # cell abbreviated: "DBATC" -> Death By A Thousand Cuts
            return by_acr[key][0]
        hits = [s for k, s in by_norm.items() if len(key) >= 8 and (k.startswith(key) or key.startswith(k))]
        return hits[0] if len(hits) == 1 else clean(cell)

    def songs(rows_, c):
        out = set()
        for r in rows_:
            cell = raw.iloc[r, c]
            if isinstance(cell, str) and clean(cell):
                out.add(match(cell))
        return out

    labels = raw[0].map(clean)
    g0, p0 = labels[labels == "Guitar"].index[0], labels[labels == "Piano"].index[0]
    guitar_rows = list(range(g0, p0))
    piano_rows = list(range(p0, start))
    guitar, piano = [songs(guitar_rows, c) for c in cols], [songs(piano_rows, c) for c in cols]
    known = set(catalogue.song)
    extra = sorted({s for show in guitar + piano for s in show if s not in known})   # covers, collaborations, soundtracks
    if extra:
        catalogue = pd.concat([catalogue, pd.DataFrame({"song": extra, "album": "OTHER", "main_set": False})],
                              ignore_index=True)
    return guitar, piano, catalogue


def load(data_dir) -> Tour:
    data_dir = Path(data_dir)
    shows, outfits = parse_outfits(data_dir / "outfits.csv")
    guitar, piano, catalogue = parse_songs(data_dir / "songs.csv", len(shows))
    return Tour(shows, outfits, guitar, piano, catalogue)
