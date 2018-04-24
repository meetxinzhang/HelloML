
import pandas as pd
import statsmodels.api as sm


df7 = pd.read_csv('source.csv')
print(df7)


y = df7.pop('Y')
X = df7

print(y)

print(X)

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.2,
                       threshold_out=0.21,
                       verbose=True):
    included = list(initial_list)
    print(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        # print(excluded)
        new_pval = pd.Series(index=excluded)
        # print(new_pval)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # print(pvalues)
        worst_pval = pvalues.max()  # null if pvalues is empty
        # print(worst_pval)
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()

            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break
    return included
pass


result = stepwise_selection(X, y)

print('resulting features:')
print(result)
