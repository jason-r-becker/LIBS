from tabulate import tabulate

def prettyPrint(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))
