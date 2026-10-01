"""Aircraft type to installed seat count, for capacity conversion.

Seat counts are not published in the ADS-B feed, and the same type flies with
markedly different layouts. Each entry therefore carries a plausible band
rather than a single number:

    seats_low    a 2-class low-density layout
    seats_typ    the most common operating layout for the type
    seats_high   a high-density leisure layout

`seats_typ` is the point estimate used for headline figures. The band is kept
in the data so downstream capacity work can propagate seat uncertainty instead
of hiding it. Values reflect published manufacturer configuration sheets and
common operator layouts; they are not operator-specific for Etihad or any
other carrier, and corporate types carry a nominal seat count because the
actual configuration is a private matter of the owner.
"""

import pandas as pd

SEATS = [
    # code, low, typical, high, family, note
    ("A124", 26, 40, 60, "An-124", "mostly freight at AUH"),
    ("A169", 6, 8, 10, "AW169", "executive, nominal seat count"),
    ("A19N", 124, 145, 156, "A320 family", "A319neo"),
    ("A300", 180, 250, 280, "A300", ""),
    ("A306", 88, 100, 120, "A300", ""),
    ("A30B", 130, 150, 180, "A300", ""),
    ("A310", 200, 250, 280, "A300", ""),
    ("A318", 112, 124, 138, "A320 family", ""),
    ("A319", 124, 138, 156, "A320 family", ""),
    ("A320", 140, 156, 180, "A320 family", ""),
    ("A321", 180, 200, 236, "A320 family", "cabin height limits seating"),
    ("A20N", 150, 174, 186, "A320 family", "A320neo"),
    ("A21N", 165, 190, 244, "A320 family", "A321neo, optional rear cabin"),
    ("A332", 240, 280, 350, "A330", ""),
    ("A333", 260, 300, 400, "A330", ""),
    ("A338", 260, 300, 400, "A330", "A330-800neo"),
    ("A339", 280, 300, 400, "A330", "A330-900neo"),
    ("A358", 300, 340, 400, "A350", "A350-1000"),
    ("A359", 280, 310, 380, "A350", ""),
    ("A35K", 275, 300, 350, "A350", "A350-900"),
    ("A388", 300, 440, 555, "A380", "capacity rides limited to Dubai"),
    ("AT43", 26, 30, 48, "ATR", "ATR 42"),
    ("AT72", 68, 72, 78, "ATR", "ATR 72"),
    ("AT76", 70, 76, 86, "ATR", "ATR 76"),
    ("B212", 8, 13, 15, "Bell 212", ""),
    ("B33N", 145, 174, 200, "B737 MAX", "737 MAX 8"),
    ("B37N", 150, 180, 200, "B737 MAX", "737 MAX 9"),
    ("B38M", 165, 210, 288, "B737 MAX", "737 MAX 8"),
    ("B39M", 178, 220, 290, "B737 MAX", "737 MAX 9"),
    ("B412", 24, 30, 36, "Bell 412", "executive, nominal seat count"),
    ("B462", 64, 70, 80, "BAe 146", ""),
    ("B463", 80, 100, 112, "BAe 146", ""),
    ("B733", 107, 126, 149, "B737", ""),
    ("B734", 130, 150, 189, "B737", "B737-400"),
    ("B735", 100, 126, 149, "B737", ""),
    ("B736", 107, 126, 149, "B737", ""),
    ("B737", 107, 126, 149, "B737", ""),
    ("B738", 159, 189, 210, "B737", "737-800"),
    ("B739", 162, 193, 220, "B737", "737-900"),
    ("B742", 268, 340, 378, "B747", "747-200"),
    ("B744", 313, 416, 471, "B747", "747-400"),
    ("B752", 160, 202, 240, "B757", ""),
    ("B753", 186, 243, 290, "B757", ""),
    ("B762", 200, 247, 290, "B767", ""),
    ("B763", 240, 290, 340, "B767", ""),
    ("B764", 260, 320, 380, "B767", ""),
    ("B772", 260, 320, 396, "B777", ""),
    ("B77L", 240, 300, 364, "B777", "777-200LR"),
    ("B77W", 300, 351, 396, "B777", "777-300ER, the AUH longhaul workhorse"),
    ("B778", 340, 396, 450, "B777", ""),
    ("B788", 208, 242, 290, "B787", ""),
    ("B789", 242, 288, 330, "B787", ""),
    ("B78X", 254, 296, 330, "B787", "787-10, longest-range 787"),
    ("C295", 56, 70, 80, "CASA C-295", ""),
    ("C56", 6, 8, 10, "Citation", ""),
    ("C56X", 6, 9, 12, "Citation", ""),
    ("C650", 8, 13, 16, "Citation", "corporate, nominal seat count"),
    ("CL60", 8, 12, 50, "CL-600", "corporate or airliner, wide range"),
    ("CN20", 30, 40, 50, "CASA CN-235", ""),
    ("CN35", 30, 44, 50, "CASA CN-235", ""),
    ("D228", 19, 22, 30, "Dornier 228", ""),
    ("DH8B", 15, 19, 25, "Dash 8", "Dash 8-100"),
    ("DH8C", 32, 37, 44, "Dash 8", "Dash 8-300"),
    ("DH8D", 32, 37, 44, "Dash 8", "Dash 8-400"),
    ("E170", 70, 80, 90, "Embraer E-Jet", ""),
    ("E190", 96, 100, 114, "Embraer E-Jet", ""),
    ("E195", 120, 132, 146, "Embraer E-Jet", "E195-E2"),
    ("E290", 80, 90, 100, "Embraer E-Jet", "E190-E2"),
    ("E295", 70, 76, 88, "Embraer E-Jet", "E195-E2"),
    ("EC35", 5, 7, 9, "EC135", "corporate, nominal seat count"),
    ("G450", 6, 10, 14, "Gulfstream", "corporate, nominal seat count"),
    ("GLEX", 12, 16, 22, "Gulfstream", "corporate, nominal seat count"),
    ("GL5T", 12, 14, 20, "Global 7500", "corporate, nominal seat count"),
    ("GL7C", 12, 14, 20, "Global 7500", "corporate, nominal seat count"),
    ("GL7T", 12, 14, 20, "Global 7500", "corporate, nominal seat count"),
    ("GLF4", 8, 12, 14, "Global 5500", "corporate, nominal seat count"),
    ("GLF6", 12, 14, 20, "Global 6000", "corporate, nominal seat count"),
    ("GA6C", 10, 12, 17, "Gulfstream G650", "corporate, nominal seat count"),
    ("GA7C", 10, 14, 19, "Gulfstream G700", "corporate, nominal seat count"),
    ("IL76", 90, 140, 180, "IL-76", "often a passenger-fit freighter"),
    ("IL96", 100, 150, 180, "IL-96", ""),
    ("JS31", 25, 29, 32, "BAe Jetstream 31", ""),
    ("LJ35", 6, 8, 10, "Learjet", ""),
    ("P180", 6, 9, 12, "Piaggio P.180", ""),
    ("SF34", 30, 34, 44, "Saab 340", ""),
    ("SU95", 98, 118, 128, "Sukhoi Superjet", ""),
]

COLS = ["actype", "seats_low", "seats_typ", "seats_high", "family", "note"]


def build():
    df = pd.DataFrame(SEATS, columns=COLS)
    if df.actype.duplicated().any():
        dupes = df.actype[df.actype.duplicated()].tolist()
        raise ValueError(f"duplicate aircraft types in SEATS: {sorted(set(dupes))}")
    return df.sort_values("actype").reset_index(drop=True)


def seats_table():
    return build().set_index("actype")


def add_seats(df, table=None):
    """Attach seat estimates to a frame with an `actype` column.

    Unknown types keep NaN rather than receiving a guessed value, so the share
    of traffic without a capacity estimate stays measurable.
    """
    if table is None:
        table = seats_table()
    out = df.copy()
    codes = out.actype.astype("string").str.strip().str.upper()
    out["seats_typ"] = codes.map(table.seats_typ)
    out["seats_low"] = codes.map(table.seats_low)
    out["seats_high"] = codes.map(table.seats_high)
    out["family"] = codes.map(table.family)
    return out


if __name__ == "__main__":
    print(build().to_string(index=False))
