import os
import cv2
import uuid
import shutil
import subprocess
import tempfile
import numpy as np


def _call_ridcp_cli(
    ridcp_root: str,
    weight_path: str,
    input_dir: str,
    output_dir: str,
    alpha: float = -21.25,
    use_weight: bool = True,
    python_exec: str = "python",
):
    script_path = os.path.join(ridcp_root, "inference_ridcp.py")

    cmd = [
        python_exec,
        script_path,
        "-i", input_dir,
        "-w", weight_path,
        "-o", output_dir,
    ]

    if use_weight:
        cmd.append("--use_weight")   
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]

    print("[RIDCP] Running:", " ".join(cmd))

   
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    subprocess.run(cmd, check=True, env=env, cwd=ridcp_root)




def dehaze(
    hazy_bgr: np.ndarray,
    ridcp_root: str = "external/RIDCP_dehazing",
    weight_relpath: str = "pretrained_models/pretrained_RIDCP.pth",
    alpha: float = -21.25,
    use_weight: bool = True,
    python_exec: str = "python",
) -> np.ndarray:
   
    if hazy_bgr is None:
        raise ValueError("hazy_bgr is None.")

    ridcp_root = os.path.abspath(ridcp_root)
    weight_path = os.path.join(ridcp_root, weight_relpath)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"RIDCP weight not found: {weight_path}")


    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        fname = f"{uuid.uuid4().hex}.png"
        in_path = os.path.join(tmp_in, fname)
        out_path = os.path.join(tmp_out, fname)

        cv2.imwrite(in_path, hazy_bgr)

        _call_ridcp_cli(
            ridcp_root=ridcp_root,
            weight_path=weight_path,
            input_dir=tmp_in,
            output_dir=tmp_out,
            alpha=alpha,
            use_weight=use_weight,
            python_exec=python_exec,
        )

        if not os.path.exists(out_path):

            outs = [f for f in os.listdir(tmp_out) if f.endswith((".png", ".jpg", ".jpeg"))]
            if not outs:
                raise RuntimeError("RIDCP did not produce any output image.")
            out_path = os.path.join(tmp_out, outs[0])

        out_bgr = cv2.imread(out_path, cv2.IMREAD_COLOR)
        if out_bgr is None:
            raise RuntimeError(f"Failed to read RIDCP output: {out_path}")

        return out_bgr
