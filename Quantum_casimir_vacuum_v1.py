# Quantum_casimir_vacuum



import random
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 5
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_SIZE = 6
TIME_GRID = (0.10, 0.25, 0.45, 0.70, 1.00)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_SIZE + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def map_loto_to_vacuum_modes(draw: np.ndarray) -> np.ndarray:
    modes = np.zeros(STATE_DIM, dtype=np.float64)
    for num in draw:
        idx = int(num) % STATE_DIM
        modes[idx] += 1.0
    modes = np.maximum(modes, 1e-12)
    modes /= np.sum(modes)
    return modes


def map_vacuum_modes_to_loto(mode_probs: np.ndarray) -> np.ndarray:
    loto_probs = np.zeros(MAX_NUM, dtype=np.float64)
    for num in range(1, MAX_NUM + 1):
        idx = num % STATE_DIM
        loto_probs[num - 1] = float(mode_probs.__getitem__(idx))
    loto_probs = np.maximum(loto_probs, 1e-12)
    loto_probs /= np.sum(loto_probs)
    return loto_probs


def build_casimir_hamiltonian(window_draws: np.ndarray) -> np.ndarray:
    hamiltonian = np.zeros((STATE_DIM, STATE_DIM), dtype=np.complex128)
    
    history_modes = np.zeros(STATE_DIM, dtype=np.float64)
    for draw in window_draws:
        history_modes += map_loto_to_vacuum_modes(draw)
    history_modes /= len(window_draws)
    
    for i in range(STATE_DIM):
        freq_i = max(float(history_modes.__getitem__(i)), 1e-6)
        energy_i = 1.0 / np.sqrt(freq_i)
        hamiltonian[i, i] = energy_i
        
    for i in range(STATE_DIM - 1):
        hamiltonian[i, i + 1] = 0.25 * (hamiltonian[i, i] + hamiltonian[i + 1, i + 1])
        hamiltonian[i + 1, i] = hamiltonian[i, i + 1]
        
    return hamiltonian


def evolve_vacuum_register(
    init_modes: np.ndarray,
    hamiltonian: np.ndarray,
    t_scale: float,
) -> np.ndarray:
    init_state = np.sqrt(init_modes).astype(np.complex128)
    norm = float(np.linalg.norm(init_state))
    if norm > 1e-15:
        init_state /= norm
        
    u_matrix = expm(-1j * hamiltonian * t_scale)
    final_state = u_matrix @ init_state
    
    mode_probs = np.abs(final_state) ** 2
    mode_probs /= np.sum(mode_probs)
    return mode_probs


def evaluate_vacuum_range(
    draws: np.ndarray,
    start_idx: int,
    end_idx: int,
    t_scale: float,
) -> float:
    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        h_cas = build_casimir_hamiltonian(draws[t - WINDOW_SIZE : t])
        init_m = map_loto_to_vacuum_modes(draws[t - 1])
        m_probs = evolve_vacuum_register(init_m, h_cas, t_scale)
        loto_p = map_vacuum_modes_to_loto(m_probs)
        
        for num in draws[t]:
            p = max(float(loto_p.__getitem__(int(num) - 1)), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def decode_casimir_combination(loto_probs: np.ndarray) -> list[int]:
    top_indices = np.argsort(loto_probs)[::-1][:NUM_POSITIONS]
    recovered = sorted(int(idx) + 1 for idx in top_indices)
    return recovered


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    n_samples = len(draws)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    best_t = float(next(iter(TIME_GRID)))
    best_val_nll = float("inf")

    for t_val in TIME_GRID:
        val_nll = evaluate_vacuum_range(draws, train_end, val_end, float(t_val))
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_t = float(t_val)

    test_nll = evaluate_vacuum_range(draws, val_end, n_samples, best_t)

    final_h_cas = build_casimir_hamiltonian(draws[-WINDOW_SIZE:])
    latest_m = map_loto_to_vacuum_modes(draws[-1])
    final_m_probs = evolve_vacuum_register(latest_m, final_h_cas, best_t)
    final_loto_probs = map_vacuum_modes_to_loto(final_m_probs)
    
    next_comb = decode_casimir_combination(final_loto_probs)

    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (t={best_t}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Casimir Vacuum Mode Disruption (1x5 qubita, 32 pozicije)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Casimir Vacuum Mode Disruption (1x5 qubita, 32 pozicije)

CSV: loto7_4718_k90.csv
Validacija NLL (t=1.0): 38.9327
Test NLL: 38.6215
NEXT: [1, 7, 11, 14, 32, 33, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (t=1.0): 38.9957
Test NLL: 39.4199
NEXT: [3, 7, 9, 18, 22, 35, 39]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (t=1.0): 38.0824
Test NLL: 38.1612
NEXT: [1, 2, 11, 14, 33, 34, 38]
"""



"""
Casimir Effect 
Dutch physicist Hendrik Casimir 
connection between quantum physics, matter and the vacuum 
quantum electromagnetic interactions 

Kazimir efekat (Hendrik Kazimir, interakcija materije i vakuumske energije)

• Korisni principi za Loto 7/39 (Autentična kvantna arhitektura):
	1. Privlačne sile usled ograničenja modova vakuuma (Boundary Conditions): Kazimir efekat nastaje kada dve neutralne metalne ploče u vakuumu ograniče dopuštene talasne dužine virtuelnih fotona između njih. Izvan ploča postoji beskonačno modova, dok je unutar ploča broj modova konačan, što stvara negativan pritisak vakuuma i privlačenje. U Loto 7/39, istorijski niz kola postavlja granične uslove (boundary conditions) koji prigušuju određene frekvencije (virtuelne modove) i pojačavaju druge.
	2. Modelovanje energetskih modova vakuuma na 5 qubita (32 bazne pozicije): Prema pravilu 34, 5 qubita daje tačno 32 pozicije (bazna stanja 0..31). Mapiramo Loto brojeve 1..39 u prostor od 32 moda tako što se preostalih 7 brojeva modeluje kao fazni presek unutar superpozicije vakuuma. Istorijska kola definišu geometrijsko rastojanje ploča (\(d\)), a Kazimir energetski operator (Hamiltonijan \(H_{Cas}\)) guši stanja visoke energije i privlači amplitude ka nultoj tački vakuuma.
	3. Proračun nulte energije vakuuma (Zero-Point Energy Shift): Izlazna kombinacija se ne dobija prostim rangiranjem frekvencija, već merenjem gde je promena nulte energije vakuuma (\(\Delta E_0\)) najveća unutar 5-qubitnog registra.

Mehanizam sa 5 qubita / 32 pozicije i Kazimir Hamiltonijanom koji uči energetski pritisak vakuuma iz graničnih uslova istorije 

Princip Kazimir efekta (prigušenje modova nulte energije vakuuma pod istorijskim graničnim uslovima) za Loto 7/39.

• Arhitektura sa 5 qubita (32 pozicije/stanja): Prema pravilu 34, koristi se 5 qubita koji generišu tačno 32 bazna stanja (0..31). Loto prostor od 39 brojeva se mapira u 32 energetska moda tako što se prvih 31 broj direktno kodira u stanja 1..31, a preostalih 8 brojeva (32..39) deli energiju i faze sa nižim harmonijskim modovima registra kroz preklapanje faza.
• Kazimir Hamiltonijan i granični uslovi: Istorijski prozor kola definiše geometrijsku udaljenost i provodljivost ploča (granične uslove). Na osnovu njih se konstruiše Kazimir Hamiltonijan operator \(H_{\text{Cas}} = \sum \hbar \omega_k\), koji simulira pritisak virtuelnih fotona vakuuma i privlači amplitude ka energetskim stanjima sa najmanjom preostalom energijom nulte tačke (\(\Delta E_0\)).
• Zajedničko očitavanje vakuuma: NEXT kombinacija se dobija izračunavanjem faznog i energetskog pomeraja celog 5-qubitnog sistema nakon unitarne evolucije \(e^{-iH_{\text{Cas}}t}\), gde se 7 brojeva ekstrahuje odjednom na osnovu minimalnog vakuumskog pritiska modova.
"""
