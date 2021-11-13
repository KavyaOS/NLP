import pandas as pd

bnc_to_bnc = pd.read_csv('output_ppl/trained-bnc_to_test-bnc.txt', delimiter='\t', header=None).mean()
bnc_to_maptask = pd.read_csv('output_ppl/trained-bnc_to_test-maptask.txt', delimiter='\t', header=None).mean()
bnc_to_switchboard = pd.read_csv('output_ppl/trained-bnc_to_test-switchboard.txt', delimiter='\t', header=None).mean()

trained_bnc = pd.concat([bnc_to_bnc, bnc_to_maptask, bnc_to_switchboard], axis=1)

maptask_to_bnc = pd.read_csv('output_ppl/trained-maptask_to_test-bnc.txt', delimiter='\t', header=None).mean()
maptask_to_maptask = pd.read_csv('output_ppl/trained-maptask_to_test-maptask.txt', delimiter='\t', header=None).mean()
maptask_to_switchboard = pd.read_csv('output_ppl/trained-maptask_to_test-switchboard.txt', delimiter='\t', header=None).mean()

trained_maptask = pd.concat([maptask_to_bnc, maptask_to_maptask, maptask_to_switchboard], axis=1)

switchboard_to_bnc = pd.read_csv('output_ppl/trained-switchboard_to_test-bnc.txt', delimiter='\t', header=None).mean()
switchboard_to_maptask = pd.read_csv('output_ppl/trained-switchboard_to_test-maptask.txt', delimiter='\t', header=None).mean()
switchboard_to_switchboard = pd.read_csv('output_ppl/trained-switchboard_to_test-switchboard.txt', delimiter='\t', header=None).mean()

trained_switchboard = pd.concat([switchboard_to_bnc, switchboard_to_maptask, switchboard_to_switchboard], axis=1)

result_mean = pd.concat([trained_bnc, trained_maptask, trained_switchboard], axis=0)

result_mean.index = result_mean.columns = ["BNC", "Maptask", "switchboard"]

print("\n\n----------------Displaying mean values------------\n\n")

print(result_mean)

bnc_to_bnc = pd.read_csv('output_ppl/trained-bnc_to_test-bnc.txt', delimiter='\t', header=None).std()
bnc_to_maptask = pd.read_csv('output_ppl/trained-bnc_to_test-maptask.txt', delimiter='\t', header=None).std()
bnc_to_switchboard = pd.read_csv('output_ppl/trained-bnc_to_test-switchboard.txt', delimiter='\t', header=None).std()

trained_bnc = pd.concat([bnc_to_bnc, bnc_to_maptask, bnc_to_switchboard], axis=1)

maptask_to_bnc = pd.read_csv('output_ppl/trained-maptask_to_test-bnc.txt', delimiter='\t', header=None).std()
maptask_to_maptask = pd.read_csv('output_ppl/trained-maptask_to_test-maptask.txt', delimiter='\t', header=None).std()
maptask_to_switchboard = pd.read_csv('output_ppl/trained-maptask_to_test-switchboard.txt', delimiter='\t', header=None).std()

trained_maptask = pd.concat([maptask_to_bnc, maptask_to_maptask, maptask_to_switchboard], axis=1)

switchboard_to_bnc = pd.read_csv('output_ppl/trained-switchboard_to_test-bnc.txt', delimiter='\t', header=None).std()
switchboard_to_maptask = pd.read_csv('output_ppl/trained-switchboard_to_test-maptask.txt', delimiter='\t', header=None).std()
switchboard_to_switchboard = pd.read_csv('output_ppl/trained-switchboard_to_test-switchboard.txt', delimiter='\t', header=None).std()

trained_switchboard = pd.concat([switchboard_to_bnc, switchboard_to_maptask, switchboard_to_switchboard], axis=1)

result_sd = pd.concat([trained_bnc, trained_maptask, trained_switchboard], axis=0)

result_sd.index = result_sd.columns = ["BNC", "Maptask", "switchboard"]

print("\n\n----------------Displaying SD values------------------\n\n")

print(result_sd)

print("\n\n")