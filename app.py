# app.py
"""
Interactive OCDMA Simulator (Streamlit)
- Supports Walsh-Hadamard (coherent) and Random Binary (incoherent) codes
- Simulates multiuser encoding/decoding (OOK intensity domain)
- Visualizes correlation matrix, spectra, eye diagram, and BER curves
"""

from functools import lru_cache
import json
import io
import numpy as np
from scipy.linalg import hadamard
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Interactive OCDMA Lab", layout="wide")

# -------------------------
# Utility / simulation code
# -------------------------
@lru_cache(maxsize=16)
def generate_hadamard(order):
    # order must be power of 2
    H = hadamard(order)
    # Convert {-1,1} to {0,1} intensity chips for incoherent intensity OOK mapping:
    return (H + 1) // 2

def generate_random_binary_codes(n_codes, length, weight=None, enforce_unique=True, max_tries=2000):
    rng = np.random.default_rng()
    codes = []
    tries = 0
    while len(codes) < n_codes and tries < max_tries:
        tries += 1
        if weight is None:
            c = rng.integers(0,2,size=length)
            # ensure not all zero
            if c.sum() == 0:
                continue
        else:
            idx = rng.choice(length, size=weight, replace=False)
            c = np.zeros(length, dtype=int)
            c[idx] = 1
        # optional uniqueness/enforce low auto-corr can be extended
        if enforce_unique and any(np.array_equal(c, e) for e in codes):
            continue
        codes.append(c)
    if len(codes) < n_codes:
        raise RuntimeError("Couldn't generate requested unique codes; try increasing length or relaxing constraints.")
    return np.array(codes)

def spread_bits(bits, code):
    # bits: shape (n_bits,) values 0/1
    # code: length L 0/1
    # return spread waveform length n_bits*L
    return np.repeat(bits, len(code)) * np.tile(code, len(bits))

def superpose_users(spread_signals, chip_rate_oversample=1):
    # simple sum across users
    return spread_signals.sum(axis=0)

def add_awgn(signal, snr_db):
    # SNR is per-chip: signal power / noise power
    sig_pow = np.mean(signal**2)
    snr_linear = 10**(snr_db/10)
    noise_pow = sig_pow / snr_linear if snr_linear > 0 else sig_pow
    noise = np.random.normal(0, np.sqrt(noise_pow), size=signal.shape)
    return signal + noise

def correlate_receiver(received, code, bit_length):
    # received: 1D array length nbits*L
    L = len(code)
    nbits = len(received) // L
    reshaped = received.reshape((nbits, L))
    # correlation = dot with code
    corr = reshaped @ code
    # matched filter threshold: half of max possible correlation (i.e., code.sum()/2)
    threshold = code.sum() / 2.0
    decisions = (corr >= threshold).astype(int)
    return decisions, corr

def compute_ber(decoded_bits, tx_bits):
    return np.mean(decoded_bits != tx_bits)

def compute_crosscorrelations(codes):
    # cross-correlation matrix (c_i dot c_j)
    return codes @ codes.T

def code_spectrum(code):
    # spectrum magnitude of code (zero mean)
    x = code - np.mean(code)
    S = np.abs(np.fft.fftshift(np.fft.fft(x, 512)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(S)))
    return freqs, S

# -------------------------
# Streamlit UI
# -------------------------
st.title("Interactive Optical CDMA (OCDMA) Lab")
st.markdown("Design codes, simulate multiuser transmission (intensity/OOK), and visualise BER, cross-correlation, spectra and eye diagrams. No datasets — fully synthetic simulation.")

# Sidebar controls
st.sidebar.header("Simulation parameters")
code_type = st.sidebar.selectbox("Code type", options=["Walsh-Hadamard (coherent)", "Random Binary (incoherent)"])
code_length = st.sidebar.selectbox("Code length (chips)", options=[8,16,32,64], index=1)
n_users = st.sidebar.slider("Number of users", min_value=1, max_value=16, value=4)
bits_per_user = st.sidebar.slider("Bits per user", min_value=10, max_value=200, value=40)
snr_db = st.sidebar.slider("SNR (dB)", min_value=-5, max_value=30, value=15)
random_weight = st.sidebar.slider("Random-code weight (if Random Binary, 0 = automatic)", min_value=0, max_value=code_length, value=0)
seed = st.sidebar.number_input("Random seed (0 = random)", value=1234)
delay_range = st.sidebar.slider("Max integer chip offset for users (0 = no delay)", min_value=0, max_value=code_length//2, value=0)
st.sidebar.write(" ")

# Buttons and small controls
col1, col2 = st.sidebar.columns([1,1])
if col1.button("Run simulation"):
    run_sim = True
else:
    run_sim = False

if col2.button("Export config"):
    config = dict(code_type=code_type, code_length=code_length, n_users=n_users, bits_per_user=bits_per_user, snr_db=snr_db, random_weight=random_weight, seed=int(seed), delay_range=delay_range)
    st.sidebar.download_button("Download config (json)", data=json.dumps(config, indent=2), file_name="ocdma_config.json")

# Default random seed handling
rng = np.random.default_rng(None if seed == 0 else int(seed))

# Generate codes
if code_type.startswith("Walsh"):
    try:
        H = generate_hadamard(code_length)
    except Exception as e:
        st.error(f"Hadamard generation failed for length {code_length}: {e}")
        st.stop()
    # Use first n_users rows (unique orthogonal codes); map to 0/1
    if n_users > H.shape[0]:
        st.warning("Number of users exceeds Hadamard order — using modulo assignment (codes will repeat).")
    codes = H[:n_users]
else:
    codes = generate_random_binary_codes(n_users, code_length, weight=(None if random_weight==0 else int(random_weight)), enforce_unique=True)

# Show codes and cross-correlation
st.subheader("Codes & cross-correlation")
colA, colB = st.columns([1,2])
with colA:
    st.write("Codes (rows = users):")
    st.dataframe(codes.astype(int), use_container_width=True)
with colB:
    cc = compute_crosscorrelations(codes)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cc, interpolation='nearest', cmap='viridis')
    ax.set_title("Cross-correlation matrix (chip dot product)")
    ax.set_xlabel("User")
    ax.set_ylabel("User")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

# Prepare transmitted bits (random)
tx_bits = rng.integers(0,2,size=(n_users, bits_per_user))

# Generate spread signals for each user, optionally apply integer chip offset
L = code_length
spreaded = np.zeros((n_users, bits_per_user * L))
for u in range(n_users):
    base = spread_bits(tx_bits[u], codes[u])
    # apply integer offset if requested
    if delay_range > 0:
        offset = rng.integers(-delay_range, delay_range+1)
    else:
        offset = 0
    if offset > 0:
        base = np.concatenate([np.zeros(offset), base])[:bits_per_user*L]
    elif offset < 0:
        base = np.concatenate([base[-offset:], np.zeros(-offset)])[:bits_per_user*L]
    spreaded[u] = base

# Superpose + noise
composite = superpose_users(spreaded)
rx = add_awgn(composite, snr_db)

# Per-user decode
decoded = np.zeros_like(tx_bits)
corr_values = []
for u in range(n_users):
    dec, corr = correlate_receiver(rx, codes[u], bits_per_user)
    decoded[u] = dec
    corr_values.append(corr)

# Compute BER per user and overall
user_bers = [compute_ber(decoded[u], tx_bits[u]) for u in range(n_users)]
overall_ber = np.mean(user_bers)

st.markdown(f"**Overall BER:** {overall_ber:.4f}    — per-user BER mean: {np.mean(user_bers):.4f}, std: {np.std(user_bers):.4f}")
st.write("Per-user BERs:")
st.bar_chart(np.array(user_bers))

# Eye diagram (overlay a few bit periods of the composite signal)
st.subheader("Eye diagram (composite received waveform overlayed per bit)")
n_overlay = min(50, bits_per_user)
fig_eye, ax_eye = plt.subplots(figsize=(6,3))
L = code_length
for i in range(n_overlay):
    segment = rx[i*L:(i+1)*L]
    ax_eye.plot(np.arange(L), segment, alpha=0.3)
ax_eye.set_xlabel("Chip index within bit")
ax_eye.set_ylabel("Amplitude")
ax_eye.set_title("Eye diagram (per-bit chip overlays)")
st.pyplot(fig_eye)

# Spectrum of selected user's code
st.subheader("Code spectrum (selected user)")
sel_user = st.selectbox("Select user to view spectrum", options=list(range(n_users)), index=0)
freqs, S = code_spectrum(codes[sel_user])
fig_sp, ax_sp = plt.subplots(figsize=(6,3))
ax_sp.plot(np.linspace(-0.5,0.5,len(S)), S)
ax_sp.set_title(f"Code spectrum (user {sel_user})")
ax_sp.set_xlabel("Normalized frequency")
ax_sp.set_ylabel("Magnitude")
st.pyplot(fig_sp)

# Correlation traces for selected user
st.subheader("Correlation trace for selected user")
fig_corr, axc = plt.subplots(figsize=(6,3))
axc.plot(corr_values[sel_user])
axc.set_title(f"Per-bit correlation values (user {sel_user})")
axc.set_xlabel("Bit index")
axc.set_ylabel("Correlation (matched filter output)")
axc.axhline(codes[sel_user].sum()/2.0, color='red', linestyle='--', label='threshold')
axc.legend()
st.pyplot(fig_corr)

# BER sweeps (optional heavy computation, cached)
st.subheader("BER sweep (SNR sweep & Users sweep)")
do_ber_sweep = st.checkbox("Compute BER vs SNR and BER vs users (may take a few seconds)", value=False)
if do_ber_sweep:
    snr_range = np.linspace(-4, 20, 10)
    users_range = list(range(1, min(12, code_length+1)))
    # BER vs SNR for fixed user count
    chosen_users = min(n_users, code_length)
    bers_snr = []
    for s in snr_range:
        # run small monte-carlo with fewer bits for speed
        B = 60
        local_tx = rng.integers(0,2,size=(chosen_users,B))
        local_spread = np.zeros((chosen_users, B * L))
        for u in range(chosen_users):
            local_spread[u] = spread_bits(local_tx[u], codes[u % len(codes)])
        composite_local = local_spread.sum(axis=0)
        rx_local = add_awgn(composite_local, s)
        decoded_local = np.zeros_like(local_tx)
        for u in range(chosen_users):
            dec, _ = correlate_receiver(rx_local, codes[u % len(codes)], B)
            decoded_local[u] = dec
        bers_snr.append(np.mean([compute_ber(decoded_local[i], local_tx[i]) for i in range(chosen_users)]))
    # plot BER vs SNR
    fig_snr, axs = plt.subplots(1,1,figsize=(6,3))
    axs.semilogy(snr_range, np.maximum(bers_snr, 1e-6), marker='o')
    axs.set_xlabel("SNR (dB)")
    axs.set_ylabel("BER (log scale)")
    axs.set_title(f"BER vs SNR (users={chosen_users})")
    st.pyplot(fig_snr)

    # BER vs number of users at current SNR
    bers_users = []
    for ucount in users_range:
        B = 80
        local_tx = rng.integers(0,2,size=(ucount,B))
        local_spread = np.zeros((ucount, B * L))
        # use available codes repeating if necessary
        for u in range(ucount):
            local_spread[u] = spread_bits(local_tx[u], codes[u % len(codes)])
        composite_local = local_spread.sum(axis=0)
        rx_local = add_awgn(composite_local, snr_db)
        decoded_local = np.zeros_like(local_tx)
        for u in range(ucount):
            dec, _ = correlate_receiver(rx_local, codes[u % len(codes)], B)
            decoded_local[u] = dec
        bers_users.append(np.mean([compute_ber(decoded_local[i], local_tx[i]) for i in range(ucount)]))
    fig_users, axu = plt.subplots(1,1,figsize=(6,3))
    axu.semilogy(users_range, np.maximum(bers_users, 1e-6), marker='o')
    axu.set_xlabel("Number of users")
    axu.set_ylabel("BER (log scale)")
    axu.set_title(f"BER vs Number of users (SNR={snr_db} dB)")
    st.pyplot(fig_users)

# Provide quick instructions for running locally or via Colab
st.markdown("---")
st.markdown("### Run instructions")
st.markdown("""
**Locally**
```bash
python -m venv .venv
source .venv/bin/activate     # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
streamlit run app.py
