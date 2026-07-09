
from agent_masked_sft.config import STANDARD_MODEL_DIR, INTERVENTIONAL_MODEL_DIR, TRUTH_PREFERENCE_FIG_PATH
from agent_masked_sft.fact_bank import FACTS
from agent_masked_sft.model import load_saved_model_and_tokenizer
from agent_masked_sft.evaluation import agent_followup, score_logprobs
from agent_masked_sft.visualisation import plot_truth_preference_deltas



STMT_W = 70
RESP_W = 80
TOTAL_W = STMT_W + 2 + RESP_W + 2 + RESP_W


def truncate(text: str, width: int) -> str:
    """
    Truncate long text so terminal tables remain readable.
    """
    text = text.replace("\n", " ")
    return text[: width - 2]


def main():
    print("Loading standard SFT model...")
    model_std, tokenizer_std = load_saved_model_and_tokenizer(
        STANDARD_MODEL_DIR,
    )

    print("Loading interventional / agent-masked SFT model...")
    model_iv, tokenizer_iv = load_saved_model_and_tokenizer(
        INTERVENTIONAL_MODEL_DIR,
    )

    score_rows = []
    followup_rows = []

    print()
    print("==== User states the TRUTH: how does each fine-tuned agent follow up? ====")
    print(
        f"{'true user statement':<{STMT_W}}  "
        f"{'std agent follow-up':<{RESP_W}}  "
        f"{'iv agent follow-up':<{RESP_W}}"
    )
    print("-" * TOTAL_W)

    for fact in FACTS:
        a_std = agent_followup(
            model=model_std,
            tokenizer=tokenizer_std,
            user_text=fact["true"],
        )

        a_iv = agent_followup(
            model=model_iv,
            tokenizer=tokenizer_iv,
            user_text=fact["true"],
        )

        sc_std = score_logprobs(
            model=model_std,
            tokenizer=tokenizer_std,
            fact=fact,
        )

        sc_iv = score_logprobs(
            model=model_iv,
            tokenizer=tokenizer_iv,
            fact=fact,
        )

        followup_rows.append(
            {
                "fact": fact,
                "type": "true",
                "user": fact["true"],
                "a_std": a_std,
                "a_iv": a_iv,
            }
        )

        score_rows.append(
            {
                "fact": fact,
                "score_std": sc_std,
                "score_iv": sc_iv,
            }
        )

        print(
            f"{truncate(fact['true'], STMT_W):<{STMT_W}}  "
            f"{truncate(a_std, RESP_W):<{RESP_W}}  "
            f"{truncate(a_iv, RESP_W):<{RESP_W}}"
        )


    print()
    print("==== User states a FALSE statement: how does each fine-tuned agent follow up? ====")
    print(
        f"{'false user statement':<{STMT_W}}  "
        f"{'std agent follow-up':<{RESP_W}}  "
        f"{'iv agent follow-up':<{RESP_W}}"
    )
    print("-" * TOTAL_W)

    for fact in FACTS:
        a_std = agent_followup(
            model=model_std,
            tokenizer=tokenizer_std,
            user_text=fact["false"],
        )

        a_iv = agent_followup(
            model=model_iv,
            tokenizer=tokenizer_iv,
            user_text=fact["false"],
        )

        followup_rows.append(
            {
                "fact": fact,
                "type": "false",
                "user_stmt": fact["false"],
                "a_std": a_std,
                "a_iv": a_iv,
            }
        )

        print(
            f"{truncate(fact['false'], STMT_W):<{STMT_W}}  "
            f"{truncate(a_std, RESP_W):<{RESP_W}}  "
            f"{truncate(a_iv, RESP_W):<{RESP_W}}"
        )




    print()
    print("==== Fixed-candidate score examples ====")

    for row in score_rows[:5]:
        fact = row["fact"]
        sc_std = row["score_std"]
        sc_iv = row["score_iv"]

        print(f"Topic: {fact['topic']}")
        print(
            f"  STD true:  {sc_std['true']:.4f} | "
            f"STD false: {sc_std['false']:.4f}"
        )
        print(
            f"  IV  true:  {sc_iv['true']:.4f} | "
            f"IV  false: {sc_iv['false']:.4f}"
        )
        print()

    n_std_prefers_true = sum(
        row["score_std"]["true"] > row["score_std"]["false"]
        for row in score_rows
    )

    n_iv_prefers_true = sum(
        row["score_iv"]["true"] > row["score_iv"]["false"]
        for row in score_rows
    )

    total = len(score_rows)

    print("==== Summary ====")
    print(
        f"Standard SFT prefers true completion for "
        f"{n_std_prefers_true} / {total} facts."
    )
    print(
        f"Interventional SFT prefers true completion for "
        f"{n_iv_prefers_true} / {total} facts."
    )



    print()
    print("==== IV model prefers FALSE on these facts ====")

    for row in score_rows:
        fact = row["fact"]
        sc_iv = row["score_iv"]

        if sc_iv["false"] > sc_iv["true"]:
            print(f"Topic: {fact['topic']}")
            print(f"  True:  {fact['true']}")
            print(f"  False: {fact['false']}")
            print(f"  IV true score:  {sc_iv['true']:.4f}")
            print(f"  IV false score: {sc_iv['false']:.4f}")
            print()



    print("==== Now plot the truth preference deltas of two models ... ====")
    plot_truth_preference_deltas(
        score_rows=score_rows,
        output_path=TRUTH_PREFERENCE_FIG_PATH,
        show=True,
    )

if __name__ == "__main__":
    main()