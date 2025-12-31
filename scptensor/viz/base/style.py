import matplotlib.pyplot as plt
import scienceplots

def setup_style() -> None:
    """Apply science plots style."""
    try:
        plt.style.use(['science', 'no-latex'])
    except Exception:
        # Fallback if scienceplots is not fully configured or latex missing
        plt.style.use('seaborn-v0_8-whitegrid')
