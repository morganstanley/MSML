import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATHS = ["results.csv"]

TRUE_POLICY_VALUE_PATH = "results_true_policy_value.csv"

USE_WHICH = {
    "q": "q",
    "w": "w",
    "dr": "dr"
}
HUE_NAMES = {
    USE_WHICH["q"]: "Q",
    USE_WHICH["w"]: "W",
    USE_WHICH["dr"]: "Orth"
}
KEEP_KEYS = list(USE_WHICH.values())
HUE_ORDER = [HUE_NAMES[USE_WHICH["q"]],
             HUE_NAMES[USE_WHICH["w"]],
             HUE_NAMES[USE_WHICH["dr"]]]
KEEP_LAMBDA = [1,2,4,8,16]

def main():
    results_df_list = []
    for path in RESULTS_PATHS:
        df = pd.read_csv(path, index_col=False)
        results_df_list.append(df)
    df_results = pd.concat(results_df_list, ignore_index=True)
    print(df_results.head())
    df_true_pv = pd.read_csv(TRUE_POLICY_VALUE_PATH, index_col=False)
    # df_true_pv = df_true_pv.set_index("lambda")
    # print(df_true_pv.head())
    # df = df_results.join(other=df_true_pv, on="lambda", how="outer")
    df = df_results[df_results["estimator"].isin(KEEP_KEYS)]
    df = df[df["lambda"].isin(KEEP_LAMBDA)]
    df_true_pv = df_true_pv[df_true_pv["lambda"].isin(KEEP_LAMBDA)]
    df["estimator"].replace(HUE_NAMES, inplace=True)

    df_robust = df.groupby(["rep_i", "lambda", "estimator"]).mean().reset_index()
    # df_robust = df
    plt.figure(figsize=(10, 2.5))
    ax = sns.boxplot(data=df_robust, x="lambda", y="est_policy_value",
                hue="estimator", gap=0.2, hue_order=HUE_ORDER)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    [ax.axvline(x, color = 'k', linestyle='--') for x in [0.5, 1.5, 2.5, 3.5, 4.5]] 
    vals = df_true_pv["true_policy_value"]
    for i, v in enumerate(vals):
        plt.hlines(v, xmin=i-0.5, xmax=i+0.5, color='r', linestyles='--')
    plt.xlim(-0.5, len(vals)-0.5)
    plt.xlabel("$\Lambda(s,a)$", fontsize=12)
    plt.ylabel("Estimated $V^-_{d_1}$", fontsize=12)
    plt.tight_layout()
    plt.savefig("results_boxplot_v2.pdf")

    # obtain MSE values
    df_join = df_robust.set_index("lambda")\
                .join(df_true_pv.set_index("lambda"))\
                .reset_index()\
                .set_index(["lambda", "estimator"])
    def get_mse(df):
        return ((df["true_policy_value"] - df["est_policy_value"]) ** 2).mean()

    def f(x):
        print(x)
        return 1
    estimator_order = {"Q": 1, "W": 2, "Orth": 3}
    df_mse = df_join.groupby(["lambda", "estimator"]).apply(get_mse)\
        .rename("mse")

    df_pvt = df_mse.reset_index().pivot(index="lambda", columns="estimator", values="mse")
    print(df_pvt)

if __name__ == "__main__":
    main()
