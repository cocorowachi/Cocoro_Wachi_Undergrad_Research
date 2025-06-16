def export(df, name='export'):
    csv = df.to_csv(f'{name}.csv', index = False) 