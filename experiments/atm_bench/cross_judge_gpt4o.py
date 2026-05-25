"""Cross-judge validation: re-judge ATM-Bench answers with GPT-4o-mini.

Loads existing answers from full1013_results.json and hard31_results.json,
re-evaluates them using GPT-4o-mini as judge, then compares with original
DeepSeek Flash judge results.

📄 Paper mapping: addresses §7 "Single judge model" limitation.
"""
import os, sys, json, time, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI

# --- Configuration ---
ECHO_KEY = "sk-XVCpuWxgD53WTzK9TupYmgCHiIqNYjIoNQ5h10AdaLRyjpLq"
JUDGE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 256
SLEEP = 0.05  # per-call delay to avoid rate limits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

JUDGE_SYSTEM_PROMPT = """You are an evaluator. Given a question, a ground-truth answer, and a system answer, determine if the system answer is CORRECT or INCORRECT.

Rules:
1. The system answer is CORRECT if it conveys the same factual information as the ground truth, even if wording differs.
2. The system answer is INCORRECT if it contradicts the ground truth, contains factual errors, or says it doesn't know when the ground truth provides an answer.
3. If the system answer partially matches but misses key information, mark it INCORRECT.
4. Reply with exactly one word: CORRECT or INCORRECT."""

client = OpenAI(api_key=ECHO_KEY, base_url="https://api.echoflow.cn/v1", timeout=60)


def judge_one(question, system_answer, ground_truth):
    """Judge a single answer. Returns (is_correct, raw_response)."""
    if not system_answer:
        return False, "EMPTY_ANSWER"
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Question: {question}\n"
                    f"Ground truth: {ground_truth}\n"
                    f"System answer: {system_answer}\n\nVerdict:"},
            ],
            temperature=TEMPERATURE,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        verdict = raw.strip().upper().rstrip(".,!;:")
        return verdict == "CORRECT", verdict
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, "API_ERROR"


def evaluate_file(input_path, label):
    """Load results JSON, re-judge all answers with GPT-4o-mini, save comparison."""
    with open(input_path) as f:
        data = json.load(f)

    details = data["details"]
    n = len(details)
    print(f"\n{'='*60}")
    print(f"Cross-judge ({JUDGE_MODEL}) on {label}: {n} QA")
    print(f"{'='*60}")

    # Track comparisons
    total = 0
    agree = 0
    results = []

    start = datetime.datetime.now()

    for i, d in enumerate(details):
        q = d["question"]
        gt = d["ground_truth"]

        # judge each of the 3 answers
        for ans_key, result_key, op_label in [
            ("fs_answer", "fs_correct", "R_T"),
            ("state_answer", "state_correct", "S_T"),
            ("traj_answer", "traj_correct", "T_T"),
        ]:
            ans = d.get(ans_key, "")
            orig_ok = d.get(result_key, False)
            new_ok, verdict = judge_one(q, ans, gt)

            total += 1
            if orig_ok == new_ok:
                agree += 1

            results.append({
                "qa_index": d["qa_index"],
                "operator": op_label,
                "original_judge": "deepseek-flash",
                "original_correct": orig_ok,
                "new_judge": JUDGE_MODEL,
                "new_correct": new_ok,
                "new_verdict": verdict,
                "agreement": orig_ok == new_ok,
            })

        if (i + 1) % 50 == 0:
            elapsed = (datetime.datetime.now() - start).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:4d}/{n}] agree={agree}/{total} ({agree/total*100:.1f}%) | "
                  f"{elapsed:.0f}s | {rate:.1f} QA/s")

        time.sleep(SLEEP)

    elapsed = (datetime.datetime.now() - start).total_seconds()
    agreement_rate = agree / total if total > 0 else 0

    out = {
        "metadata": {
            "experiment": f"Cross-judge validation on {label}",
            "original_judge": "deepseek-flash",
            "new_judge": JUDGE_MODEL,
            "n_qa": n,
            "n_judgments": total,
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "agreement_rate": round(agreement_rate, 4),
        "details": results,
    }

    out_path = os.path.join(RESULTS_DIR, f"cross_judge_{label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — {label}")
    print(f"  Total judgments: {total}")
    print(f"  Agreement: {agree}/{total} ({agreement_rate*100:.1f}%)")
    print(f"  Saved to: {out_path}")
    print(f"{'='*60}")

    return out


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r1 = evaluate_file(
        os.path.join(RESULTS_DIR, "full1013_results.json"),
        "full1013",
    )
    r2 = evaluate_file(
        os.path.join(RESULTS_DIR, "hard31_results.json"),
        "hard31",
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  full1013: {r1['agreement_rate']*100:.1f}% agreement "
          f"(n={r1['metadata']['n_judgments']})")
    print(f"  hard31:   {r2['agreement_rate']*100:.1f}% agreement "
          f"(n={r2['metadata']['n_judgments']})")
