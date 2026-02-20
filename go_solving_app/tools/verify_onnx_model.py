#!/usr/bin/env python3
"""
Verify the converted ONNX model on 13×13 and 19×19 boards.

Checks:
  1. 19×19 empty board: policy should be reasonable, value should not be 100% white
  2. 13×13 empty board (padded to 19×19): same checks (filtered to on-board moves)
"""

import numpy as np
import onnxruntime as ort
import sys


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run_test(session, board_size, label):
    print(f"\n{'='*60}")
    print(f"Testing {label}: {board_size}×{board_size} board (padded to 19×19)")
    print(f"{'='*60}")

    pad_size = 19
    bin_features = 22
    global_features = 19

    # Build input_binary [1, 22, 19, 19]
    input_binary = np.zeros((1, bin_features, pad_size, pad_size), dtype=np.float32)
    # Channel 0: board mask — 1 only within board_size×board_size
    input_binary[0, 0, :board_size, :board_size] = 1.0
    # Channel 8: black to play
    input_binary[0, 8, :board_size, :board_size] = 1.0

    # Build input_global [1, 19]
    input_global = np.zeros((1, global_features), dtype=np.float32)
    input_global[0, 0] = 7.5 / 20.0  # komi
    input_global[0, 1] = 1.0          # black to play

    policy, value, ownership = session.run(
        None,
        {'input_binary': input_binary, 'input_global': input_global},
    )

    # Policy analysis — filter to on-board moves + pass
    policy_flat = policy[0]
    pass_idx = pad_size * pad_size  # 361

    # Collect valid indices: positions within board_size×board_size + pass
    valid_indices = []
    for r in range(board_size):
        for c in range(board_size):
            valid_indices.append(r * pad_size + c)
    valid_indices.append(pass_idx)

    valid_logits = policy_flat[valid_indices]
    valid_probs = softmax(valid_logits)

    # Top 5 on-board moves
    top_local = np.argsort(valid_probs)[::-1][:5]
    print(f"\nPolicy (top 5 ON-BOARD moves):")
    for li in top_local:
        idx = valid_indices[li]
        if idx == pass_idx:
            move = "PASS"
        else:
            r, c = divmod(idx, pad_size)
            move = f"({r},{c})"
        print(f"  {move}: logit={policy_flat[idx]:.4f}, prob={valid_probs[li]*100:.2f}%")

    # Check: policy should have reasonable probabilities
    max_prob = valid_probs.max()
    print(f"\nMax on-board probability: {max_prob*100:.2f}%")
    if max_prob < 1e-10:
        print("  *** FAIL: All probabilities near zero — pooling bug likely present!")
    elif max_prob > 0.99:
        print("  *** WARN: One move dominates (>99%) — may be unreasonable for empty board")
    else:
        print("  OK: Probabilities are reasonable")

    # Value analysis
    print(f"\nValue output (raw): {value[0]}")
    val_probs = softmax(value[0])
    print(f"  Win(B)={val_probs[0]*100:.1f}%, Win(W)={val_probs[1]*100:.1f}%, NoResult={val_probs[2]*100:.4f}%")

    if val_probs[1] > 0.99:
        print("  *** FAIL: 100% white win — value head is broken!")
    elif val_probs[0] > 0.99:
        print("  *** WARN: 100% black win on empty board — unusual")
    else:
        print("  OK: Value distribution is reasonable")

    # Ownership analysis
    own = ownership[0, 0, :board_size, :board_size]
    print(f"\nOwnership ({board_size}×{board_size} region):")
    print(f"  mean={own.mean():.4f}, min={own.min():.4f}, max={own.max():.4f}")

    return True


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "assets/models/katago_b6c64.onnx"
    print(f"Loading model: {model_path}")

    session = ort.InferenceSession(model_path)

    run_test(session, 19, "19×19 (full board)")
    run_test(session, 13, "13×13 (padded)")
    run_test(session, 9, "9×9 (padded)")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
