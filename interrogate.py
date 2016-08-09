import pandas as pd

def mix_dataframes(df_1, df_2, group_size):
	mix_df_1 = pd.concat([df_1, df_2]).reset_index(drop=True)
	mix_df_1_gpby = mix_df_1.groupby(list(mix_df_1.columns))

	set_1 = [x[0] for x in mix_df_1_gpby.groups.values() if len(x) == group_size]
	set_1 = mix_df_1.reindex(set_1)
	return set_1

old_dictionary = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']), 'two': pd.Series([2,3,4], index=['a', 'b', 'c'])}
new_dictionary = {'one': pd.Series([2, 3, 4], index=['a', 'b', 'c']), 'two': pd.Series([3,4,5], index=['a', 'b', 'c'])}

old_df = pd.DataFrame(old_dictionary)
new_df = pd.DataFrame(new_dictionary)

set_1 = mix_dataframes(old_df, new_df, 1)
print(set_1)

set_2 = mix_dataframes(old_df, set_1, 1)
print(set_2)

set_3 = mix_dataframes(set_1, set_2, 2)
print(set_3)



