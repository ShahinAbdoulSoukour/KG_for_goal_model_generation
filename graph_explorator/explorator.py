import pandas as pd

from utils.functions import get_neighbors, test_entailment


def graph_explorator(df, goal, graph, tokenizer_nli, model_nli):
    entailed_triples_df = pd.DataFrame(columns=["SUBGOALS", "SUBGOALS_SERIALIZED", "SCORE", "NLI_LABEL"])

    for _, row in df[["PREMISE", "HYPOTHESIS", "PREMISE_SERIALIZED", "ENTAILMENT", "NLI_LABEL"]].iterrows():
        # entailment score
        entailment_score = row["ENTAILMENT"]

        if row['NLI_LABEL'] == "ENTAILMENT":
            entailed_triples_dict = {
                "SUBGOALS": row["PREMISE"],
                "SUBGOALS_SERIALIZED": row["PREMISE_SERIALIZED"],
                "SCORE": entailment_score,
                "NLI_LABEL": row["NLI_LABEL"]
            }

            # update the dataframe
            entailed_triples_df.loc[len(entailed_triples_df)] = entailed_triples_dict
        else:
            # extract triple neighbors
            neighbor_triples = get_neighbors(row["PREMISE"], row["PREMISE_SERIALIZED"], graph)

            # concatenate triples
            concatenated_triples_dict = {
                "PREMISE": neighbor_triples["TRIPLE_NEIGHBOR"] + ". " + neighbor_triples['TRIPLE'],
                "HYPOTHESIS": goal,
                "PREMISE_SERIALIZED": neighbor_triples["TRIPLE_NEIGHBOR_SERIALIZED"] + neighbor_triples['TRIPLE_SERIALIZED']
            }
            concatenated_triples_df = pd.DataFrame(concatenated_triples_dict)

            # --> test the entailment
            entailment_concatenate_triples_result = test_entailment(concatenated_triples_df, tokenizer_nli, model_nli)

            # extract the highest score (entailment)
            entailment_concatenate_triples_best_score = entailment_concatenate_triples_result.head(1)

            # check if the label is "ENTAILMENT"
            contains_entailment = any(entailment_concatenate_triples_best_score['NLI_LABEL'] == 'ENTAILMENT')

            if contains_entailment:
                # for index, row in entailment_concatenate_triples_best_score.iterrows():
                # if all(all(elem not in sublist for elem in row['PREMISE_SERIALIZED']) for sublist in entailed_triples_df['SUBGOALS_SERIALIZED']):
                entailment_concatenate_triples_best_score = entailment_concatenate_triples_best_score[
                    ['PREMISE', 'PREMISE_SERIALIZED', 'ENTAILMENT', 'NLI_LABEL']]
                entailment_concatenate_triples_best_score.rename(
                    columns={'PREMISE': 'SUBGOALS', 'PREMISE_SERIALIZED': 'SUBGOALS_SERIALIZED', 'ENTAILMENT': 'SCORE'},
                    inplace=True)

                # update the dataframe
                entailed_triples_df = pd.concat([entailed_triples_df, entailment_concatenate_triples_best_score])
            else:
                while True:
                    current_score = entailment_concatenate_triples_result.iloc[0]['ENTAILMENT']

                    # check the evolution of the entailment score
                    if current_score > entailment_score:
                        # extract the triple neighbors
                        neighbor_triples = get_neighbors(entailment_concatenate_triples_best_score.iloc[0]["PREMISE"],
                                                         entailment_concatenate_triples_best_score.iloc[0]["PREMISE_SERIALIZED"],
                                                         graph)

                        if not neighbor_triples.empty:
                            # concatenate triples
                            concatenated_triples_dict = {
                                "PREMISE": neighbor_triples["TRIPLE_NEIGHBOR"] + ". " + neighbor_triples['TRIPLE'],
                                "HYPOTHESIS": goal,
                                "PREMISE_SERIALIZED": neighbor_triples["TRIPLE_NEIGHBOR_SERIALIZED"] + neighbor_triples['TRIPLE_SERIALIZED']
                            }
                            concatenated_triples_df = pd.DataFrame(concatenated_triples_dict)

                            # --> test the entailment
                            entailment_concatenate_triples_result = test_entailment(concatenated_triples_df, tokenizer_nli, model_nli)

                            # update the previous score
                            entailment_score = current_score

                            if any(entailment_concatenate_triples_result['NLI_LABEL'] == 'ENTAILMENT'):
                                entailment_concatenate_triples_result = entailment_concatenate_triples_result.loc[
                                    entailment_concatenate_triples_result['NLI_LABEL'] == 'ENTAILMENT'][['PREMISE', 'PREMISE_SERIALIZED', 'ENTAILMENT', 'NLI_LABEL']]
                                entailment_concatenate_triples_result.rename(columns={'PREMISE': 'SUBGOALS', 'PREMISE_SERIALIZED': 'SUBGOALS_SERIALIZED', 'ENTAILMENT': 'SCORE'}, inplace=True)
                                # update the dataframe
                                entailed_triples_df = pd.concat([entailed_triples_df, entailment_concatenate_triples_result])
                                break
                            
                        else:
                            break
                    else:
                        break

    return entailed_triples_df