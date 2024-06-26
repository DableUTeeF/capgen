import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('kaggle/solution.csv')

    use = df['Usage']
    use2 = []
    for i, u in enumerate(use):
        if i < 32000 and u == 'Private':
            use2.append('Ignored')
        else:
            use2.append(u)
    df['Usage'] = use2
    df.to_csv('k2/solution.csv', index=False)

