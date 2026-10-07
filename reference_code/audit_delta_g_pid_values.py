import json
import numpy as np
import generate_elc_external_candidate_model as elc
from delta_g_pid import delta_g_pid

data = elc.simulate(seed=46, n_lineages=70, n_tau=48)
arr = elc.build_information_arrays(
    data, target_history_order=2, residual_history_order=6, within_k=2, source_k=None
)
M = arr["future"]; X = arr["current"]; Y = arr["source"]
pid = delta_g_pid(M, X, Y, rank_transform=True, bias_correct=True, max_iter=250)
h0 = elc.gc_cond_entropy_bits(M, X)
h1 = elc.gc_cond_entropy_bits(M, np.hstack([X, Y]))
before = elc.gc_cmi_bits(M, arr["older"], X)
after = elc.gc_cmi_bits(M, arr["older"], np.hstack([X, Y]))
print(json.dumps({
    "H_future_given_current_ELC_bits": h0,
    "H_future_given_current_ELC_and_ecology_bits": h1,
    "entropy_reduction_percent": 100 * max(0, h0 - h1) / h0,
    "residual_prior_history_reduction_percent": 100 * max(0, before - after) / before,
    "SI_deltaG_shared_redundant_bits": pid["RI"],
    "UI_ELC_bits": pid["UI_X"],
    "UI_eco_bits": pid["UI_Y"],
    "CI_deltaG_complementary_synergistic_bits": pid["SI"],
    "I_future_current_ELC_bits": pid["I_MX"],
    "I_future_ecology_bits": pid["I_MY"],
    "I_future_joint_sources_bits": pid["I_MXY"],
    "n": pid["n"],
    "dims": pid["dims"],
}, indent=2))
