"""SplitMix64 (Vigna): kleiner, gut gemischter 64-Bit-Zufallsgenerator in reiner Ganzzahl-Arithmetik.

numpy und `random` garantieren keine versionsstabilen Zufallsströme, die CI installiert die neueste Version; SplitMix64 liefert überall dieselbe Zahlenfolge.
Alle Zufallsentscheidungen der Demo (Flussmatrizen, Platzlagen) laufen über diese Klasse."""

_MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & _MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Ganzzahl in 0..n-1 (Modulo-Verzerrung bei n <= 1000 liegt unter 1e-16)."""
        return self.next() % n

    def randint(self, lo: int, hi: int) -> int:
        """Ganzzahl in lo..hi, beide Grenzen inklusive."""
        return lo + self.below(hi - lo + 1)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next() >> 11) / float(1 << 53)

    def choice(self, items):
        return items[self.below(len(items))]

    def shuffle(self, items: list) -> None:
        """Fisher-Yates, an Ort und Stelle."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def weighted(self, values, weights):
        """Ein Wert aus `values` mit Wahrscheinlichkeit proportional zu `weights` (ganzzahlig)."""
        r = self.below(sum(weights))
        for value, w in zip(values, weights):
            if r < w:
                return value
            r -= w
        return values[-1]
