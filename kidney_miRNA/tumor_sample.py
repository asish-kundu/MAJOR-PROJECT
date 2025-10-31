import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clinical_df = pd.read_csv('clinical.tsv', sep='\t', usecols=['primary_diagnosis', 'case_submitter_id'])
gdc_df = pd.read_csv('gdc_sample_sheet.2024-04-08.tsv', sep='\t', usecols=['Case ID'])

common_ids = clinical_df[clinical_df['case_submitter_id'].isin(gdc_df['Case ID'])].drop_duplicates(subset=['case_submitter_id'])
common_ids.to_csv('Output.csv', index=False)
