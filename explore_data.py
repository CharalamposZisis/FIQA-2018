from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent/ "fiqa"

corpus = pd.read_parquet(Path(DATA_DIR) / "corpus.parquet")
print(f" The corpus data set is: \n {corpus}")

qrels = pd.read_parquet(Path(DATA_DIR) / "qrels.parquet")
print(f" The qrels data are : \n {qrels}.")

queries = pd.read_parquet(Path(DATA_DIR) / "queries.parquet")
print(f"The queries data set is: \n {queries}.")


# print the first five queries 
print(f"The queries data are: \n {queries.text.loc[0:5]}")

# print the first five queries 
print(f"The corpus data are: \n {corpus.text.loc[0:5]}")

# print the first five qrels 
print(f"The qrels first five lines are: \n {qrels.loc[0:5]}")

# --------------------------------------------------------------
# Step 3: Inspect one query and its relevant docs
# --------------------------------------------------------------
test_query_ids = set(qrels["query-id"].astype(str))
test_queries = queries[queries["_id"].astype(str).isin(test_query_ids)]
print(f"Queries (test, with qrels): {len(test_queries)}")

query = test_queries.iloc[0]
print("\nExample query")
print(f"  _id:  {query['_id']}")
print(f"  text: {query['text']}")

relevant = qrels[qrels["query-id"].astype(str) == str(query["_id"])]
print(f"\n  Relevant docs for this question: {list(relevant['corpus-id'])}")