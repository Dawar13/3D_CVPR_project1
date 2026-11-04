"""Progress of VM session A (scripts/session_queue_e2x.sh): one bar per job. Run on the VM from ~/x/MapAnything_Exp1:
  python scripts/session_progress_e2x.py
Counting and bars are those of scripts/session_progress.py."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_progress import bar, done, expected  # noqa: E402

T = os.path.expanduser("~/x/MapAnything_Exp1")
CAL = "data/clips_e2x_*/cal__*.npz"
RD, RE = "data/clips_e2x_room/dev__*.npz", "data/clips_e2x_room/eval__*.npz"
JOBS = [("MA E3-cal synth ETH3D", "configs/e2x_e3cal_synth_mapanything.json", "data/clips_e2/dev__*__c0.npz", "results/e2x_e3cal_synth_ma_eth3d"),
        ("MA E3-cal synth cal", "configs/e2x_e3cal_synth_mapanything.json", CAL, "results/e2x_e3cal_synth_ma_cal"),
        ("MA E3-cal real cal", "configs/e2x_e3cal_real_mapanything.json", CAL, "results/e2x_e3cal_real_ma_cal"),
        ("MA E3-cal real roomdev", "configs/e2x_e3cal_real_mapanything.json", RD, "results/e2x_e3cal_real_ma_roomdev"),
        ("MA room dev", "configs/e2x_real_mapanything.json", RD, "results/e2x_room_ma_dev"),
        ("MA room eval", "configs/e2x_real_mapanything.json", RE, "results/e2x_room_ma_eval"),
        ("PX checks", "configs/e2x_pi3x_checks.json", "data/clips/dev__*.npz", "results/e2x_pi3x_checks"),
        ("PX ETH3D dev", "configs/e2x_pi3x_eth3d.json", "data/clips_e2/dev__*__c0.npz", "results/e2x_pi3x_eth3d_dev"),
        ("PX ETH3D eval", "configs/e2x_pi3x_eth3d.json", "data/clips_e2/eval__*.npz", "results/e2x_pi3x_eth3d_eval"),
        ("PX E2-E dev", "configs/e2x_real_pi3x.json", "data/clips_e2e/dev__*.npz", "results/e2x_e2e_px_dev"),
        ("PX E2-E eval", "configs/e2x_real_pi3x.json", "data/clips_e2e/eval__*.npz", "results/e2x_e2e_px_eval"),
        ("PX room dev", "configs/e2x_real_pi3x.json", RD, "results/e2x_room_px_dev"),
        ("PX room eval", "configs/e2x_real_pi3x.json", RE, "results/e2x_room_px_eval"),
        ("PX E3-cal synth ETH3D", "configs/e2x_e3cal_synth_pi3x.json", "data/clips_e2/dev__*__c0.npz", "results/e2x_e3cal_synth_px_eth3d"),
        ("PX E3-cal synth cal", "configs/e2x_e3cal_synth_pi3x.json", CAL, "results/e2x_e3cal_synth_px_cal"),
        ("PX E3-cal real cal", "configs/e2x_e3cal_real_pi3x.json", CAL, "results/e2x_e3cal_real_px_cal"),
        ("OV room dev", "configs/e2x_real_omnivggt.json", RD, "results/e2x_room_ov_dev"),
        ("OV room eval", "configs/e2x_real_omnivggt.json", RE, "results/e2x_room_ov_eval"),
        ("OV E3-cal synth ETH3D", "configs/e2x_e3cal_synth_omnivggt.json", "data/clips_e2/dev__*__c0.npz", "results/e2x_e3cal_synth_ov_eth3d"),
        ("OV E3-cal synth cal", "configs/e2x_e3cal_synth_omnivggt.json", CAL, "results/e2x_e3cal_synth_ov_cal"),
        ("OV E3-cal real cal", "configs/e2x_e3cal_real_omnivggt.json", CAL, "results/e2x_e3cal_real_ov_cal")]

if __name__ == "__main__":
    tot_d = tot_t = 0
    for name, cfg, clips, out in JOBS:
        d, nf = done(T, out)
        t = expected(T, cfg, clips)
        tot_d, tot_t = tot_d + d, tot_t + t
        print(bar(name, d, t, w=24) + (f"   FAILED: {nf}" if nf else ""))
    print(bar("ALL", tot_d, tot_t, w=24))
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
