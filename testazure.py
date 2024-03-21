import mlflow
import pandas as pd

tracking_uri = "http://localhost:5000/"
mlflow.set_tracking_uri(tracking_uri)

eval_data = pd.DataFrame(
    {
        "inputs": [
            "Write a report on Covid 19 on health care in India",
            "check the grammar for I very intested in working in as a software engineer in kryptos technologies ",
        ],
        "ground_truth": [
            "Here is the report on the impact of COVID-19 on healthcare in India:The COVID-19 pandemic has had a significant impact on healthcare in India. The healthcare system faced a strain on resources, including a lack of oxygen and essential drugs required for the treatment of COVID-19. This led to a reconfiguration of care in hospitals, with many patients suffering non-COVID conditions having to delay their treatment. Furthermore, there was a drastic decline in seeking non-COVID-19 disease related healthcare services during the pandemic period. Despite these challenges, India demonstrated a high proportion of recovered COVID-19 cases, which was testimony to the technological advances applied at scale against the pandemic.",
            "I am very interested in working as a software engineer at Kryptos Technologies",
        ],
        "predictions": [
            "The surge in COVID-19 cases strained India's healthcare infrastructure, leading to shortages of hospital beds, medical oxygen, and essential supplies. Overwhelmed hospitals struggled to accommodate patients, exacerbating the crisis.",
            "I am very interested in working as a software engineer at Kryptos Technologies."
            
        ]
    }
)

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        data=eval_data,
        targets="ground_truth",
        predictions="predictions",
        extra_metrics=[mlflow.metrics.genai.answer_similarity()],
        evaluators="default",
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    eval_table = results.tables["eval_results_table"]
    print(f"See evaluation table below: \n{eval_table}")
    # eval_table.to_excel("eval_results.xlsx", index=False)