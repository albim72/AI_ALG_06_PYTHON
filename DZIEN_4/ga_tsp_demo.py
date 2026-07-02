"""
ALGORYTM GENETYCZNY — Problem Komiwojażera (TSP)
=================================================
Demo szkoleniowe: ewolucja trasy przez 50 miast.

Dlaczego TSP?
  - przestrzeń rozwiązań: 50! ≈ 3 × 10^64 permutacji (brute force niemożliwy)
  - wynik widać GOŁYM OKIEM: chaos splątanych linii -> czysta pętla
  - każdy operator GA ma tu naturalną interpretację

Struktura (klasyczny cykl ewolucyjny):
  1. Reprezentacja  — chromosom = permutacja miast
  2. Fitness        — długość trasy (minimalizujemy)
  3. Selekcja       — turniejowa (k osobników walczy, wygrywa najlepszy)
  4. Krzyżowanie    — Ordered Crossover (OX): zachowuje permutację
  5. Mutacja        — inwersja segmentu (2-opt-like, silniejsza niż swap)
  6. Elityzm        — najlepsi przechodzą bez zmian

Uruchomienie:  python ga_tsp_demo.py
Wynik:         ga_tsp_wynik.png + log w konsoli + porównanie z random search
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # zmień na "TkAgg" dla animacji na żywo podczas szkolenia
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# PARAMETRY — na szkoleniu warto je zmieniać NA ŻYWO i pokazywać skutki
# ----------------------------------------------------------------------
N_MIAST        = 50
POPULACJA      = 300
GENERACJE      = 600
TURNIEJ_K      = 5       # większe k = silniejsza presja selekcyjna
P_KRZYZOWANIA  = 0.9
P_MUTACJI      = 0.25    # prawdopodobieństwo mutacji osobnika
ELITA          = 4       # ilu najlepszych przechodzi bez zmian


# ----------------------------------------------------------------------
# 1. ŚWIAT: losowe miasta + macierz odległości (policzona raz)
# ----------------------------------------------------------------------
miasta = rng.random((N_MIAST, 2)) * 100
DIST = np.sqrt(((miasta[:, None, :] - miasta[None, :, :]) ** 2).sum(-1))


# ----------------------------------------------------------------------
# 2. FITNESS — wektorowo dla całej populacji naraz
# ----------------------------------------------------------------------
def dlugosc_tras(pop: np.ndarray) -> np.ndarray:
    """pop: (n_osobnikow, n_miast) -> długości zamkniętych tras."""
    nastepne = np.roll(pop, -1, axis=1)
    return DIST[pop, nastepne].sum(axis=1)


# ----------------------------------------------------------------------
# 3. SELEKCJA TURNIEJOWA
# ----------------------------------------------------------------------
def selekcja_turniejowa(pop, fitness, n_wybranych):
    """Losuj k zawodników, wybierz najlepszego. Powtórz n razy."""
    kandydaci = rng.integers(0, len(pop), size=(n_wybranych, TURNIEJ_K))
    zwyciezcy = kandydaci[np.arange(n_wybranych),
                          fitness[kandydaci].argmin(axis=1)]
    return pop[zwyciezcy]


# ----------------------------------------------------------------------
# 4. KRZYŻOWANIE OX (Ordered Crossover)
#    Wycinamy segment z rodzica A, resztę uzupełniamy kolejnością z B.
#    Kluczowe: wynik ZAWSZE jest poprawną permutacją.
# ----------------------------------------------------------------------
def krzyzowanie_ox(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    dziecko = np.full(n, -1)
    dziecko[i:j] = a[i:j]
    reszta = b[~np.isin(b, dziecko[i:j])]
    dziecko[np.concatenate([np.arange(j, n), np.arange(0, i)])] = reszta
    return dziecko


# ----------------------------------------------------------------------
# 5. MUTACJA PRZEZ INWERSJĘ — odwróć losowy segment trasy.
#    Geometrycznie: "rozplątanie" jednego skrzyżowania linii.
# ----------------------------------------------------------------------
def mutacja_inwersja(osobnik: np.ndarray) -> np.ndarray:
    i, j = sorted(rng.choice(len(osobnik), size=2, replace=False))
    osobnik[i:j] = osobnik[i:j][::-1]
    return osobnik


# ----------------------------------------------------------------------
# GŁÓWNA PĘTLA EWOLUCJI
# ----------------------------------------------------------------------
def ewolucja():
    pop = np.array([rng.permutation(N_MIAST) for _ in range(POPULACJA)])
    historia_best, historia_avg, snapshoty = [], [], {}

    for gen in range(GENERACJE):
        fit = dlugosc_tras(pop)
        porzadek = fit.argsort()
        pop, fit = pop[porzadek], fit[porzadek]

        historia_best.append(fit[0])
        historia_avg.append(fit.mean())
        if gen in (0, 20, 100, GENERACJE - 1):
            snapshoty[gen] = (pop[0].copy(), fit[0])
        if gen % 100 == 0:
            print(f"  gen {gen:4d} | best {fit[0]:8.2f} | avg {fit.mean():8.2f}")

        # elityzm
        nowa = [pop[k].copy() for k in range(ELITA)]

        # reprodukcja
        rodzice = selekcja_turniejowa(pop, fit, 2 * (POPULACJA - ELITA))
        for k in range(0, len(rodzice) - 1, 2):
            a, b = rodzice[k], rodzice[k + 1]
            dziecko = krzyzowanie_ox(a, b) if rng.random() < P_KRZYZOWANIA else a.copy()
            if rng.random() < P_MUTACJI:
                dziecko = mutacja_inwersja(dziecko)
            nowa.append(dziecko)

        pop = np.array(nowa[:POPULACJA])

    fit = dlugosc_tras(pop)
    best = pop[fit.argmin()]
    return best, fit.min(), historia_best, historia_avg, snapshoty


# ----------------------------------------------------------------------
# BASELINE: random search z TYM SAMYM budżetem ewaluacji
#    Kluczowy argument dydaktyczny: GA to nie "losowanie z fartem".
# ----------------------------------------------------------------------
def random_search(budzet: int) -> float:
    najlepszy = np.inf
    for _ in range(budzet // 10_000):
        pop = np.array([rng.permutation(N_MIAST) for _ in range(10_000)])
        najlepszy = min(najlepszy, dlugosc_tras(pop).min())
    return najlepszy


# ----------------------------------------------------------------------
# WIZUALIZACJA: 4 fazy ewolucji + krzywa zbieżności
# ----------------------------------------------------------------------
def rysuj(best, best_len, hist_best, hist_avg, snapshoty, rs_len):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Algorytm genetyczny — TSP, 50 miast, przestrzeń 3×10⁶⁴ rozwiązań",
                 fontsize=15, fontweight="bold")

    for idx, (gen, (trasa, dl)) in enumerate(sorted(snapshoty.items())):
        ax = fig.add_subplot(2, 4, idx + 1)
        pts = miasta[np.append(trasa, trasa[0])]
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.2,
                color=plt.cm.viridis(idx / 3))
        ax.scatter(miasta[:, 0], miasta[:, 1], s=18, c="crimson", zorder=3)
        ax.set_title(f"Generacja {gen}  |  trasa = {dl:.1f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(hist_best, lw=2, label="najlepszy osobnik")
    ax.plot(hist_avg, lw=1, alpha=0.6, label="średnia populacji")
    ax.axhline(rs_len, color="gray", ls="--",
               label=f"random search, ten sam budżet ({rs_len:.1f})")
    ax.set_xlabel("generacja"); ax.set_ylabel("długość trasy")
    ax.set_title("Zbieżność ewolucji vs losowe przeszukiwanie")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ga_tsp_wynik.png", dpi=130)
    print("\nZapisano wykres: ga_tsp_wynik.png")


if __name__ == "__main__":
    print("Ewolucja startuje…")
    best, best_len, hb, ha, snap = ewolucja()

    budzet = POPULACJA * GENERACJE
    print(f"\nRandom search (budżet: {budzet:,} ewaluacji)…")
    rs = random_search(budzet)

    poprawa = (rs - best_len) / rs * 100
    print(f"\n{'='*50}")
    print(f"  GA:            {best_len:8.2f}")
    print(f"  Random search: {rs:8.2f}")
    print(f"  GA lepszy o:   {poprawa:8.1f}%")
    print(f"{'='*50}")

    rysuj(best, best_len, hb, ha, snap, rs)
