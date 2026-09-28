# Exchange Simulator

A two-state chemical exchange simulator, originally built when I was writing my thesis to better understand the various chemical exchange experiments I was doing during my PhD.

The repository contains a Streamlit app to explore how CPMG, CEST, transverse relaxation, R1rho, and $c_{fast}$ profiles change with different two-site exchange parameters. The R1rho experiments simulated here are constant-time R1rho experiments originally reported in Yuwen et al. ([2018](https://pubmed.ncbi.nlm.nih.gov/29303268/)).

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib
- SymPy
- Streamlit

## Installation

From the repository root:

```bash
python -m pip install -e .
```

If you use a conda environment, activate it before installing and running the application:

```bash
conda activate nmr_env
python -m pip install -e .
```

## Run the app

```bash
streamlit run simulate.py
```

The app opens a browser dashboard with all simulated profiles displayed together in six panels:

1. CPMG relaxation dispersion
2. CEST profiles
3. Transverse relaxation profiles for states A, B, and the observed signal
4. Transverse relaxation rate bar chart
5. R1rho profile
6. cfast profile

Use the **Simulate profiles** button after changing parameters.

## Parameters

### Exchange parameters

- `Delta omega`: frequency difference between states A and B, in Hz
- `Delta R2`: difference used to derive `R2B`, in s-1
- `R2A`, `R1A`: relaxation rates for state A, in s-1
- `Population B`: population fraction of state B
- `kex`: exchange rate, in s-1
- `omegaA`: resonance frequency of state A, in Hz
- `B0`: spectrometer field, in MHz

The app derives:

```text
R2B = R2A + Delta R2
omegaB = omegaA + Delta omega
```

### Experiment-specific settings

Each experiment has its own relaxation time and RF field:

- CPMG: `CPMG T_relax` and `CPMG RF field omega1`
- CEST: `CEST T_relax` and selectable B1 fields
- R1rho: `R1rho T_relax` and `R1rho RF field omega1`

The CEST B1 selector defaults to 250, 500, and 750 Hz. Multiple CEST curves are plotted together and labeled by B1 field.

## Project layout

```text
simulate.py                         Streamlit application
setup.py                            Package metadata and dependencies
exchange_simulator/simulators/      Numerical simulation implementations
exchange_simulator/data_reader/     Input data readers
exchange_simulator/fitters/         Fitting utilities
notebooks/                          Exploratory notebooks
```

## Notes

The public simulator inputs use frequencies in Hz. The underlying Liouvillian implementations convert frequencies to angular frequency internally where required.
