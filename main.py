# Step 1: Load all homework files from the 'data' folder
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_homeworks(folder):
    files = os.listdir(folder)
    texts = {}
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts[file] = f.read()
    return texts

homeworks = load_homeworks("data")
names = list(homeworks.keys())
texts = list(homeworks.values())

# Step 2: Convert homework text into embeddings using a pre-trained AI model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

print(f"Generated embeddings for {len(embeddings)} files.")

# Step 3: Compare embeddings to detect plagiarism
similarity_matrix = cosine_similarity(embeddings)
threshold = 0.75  # You can adjust this threshold as needed

print("\n🔍 Potentially plagiarized pairs:")
plagiarized_pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        similarity = similarity_matrix[i][j]
        if similarity > threshold:
            print(f"⚠️  {names[i]} and {names[j]} are similar! Similarity Score: {similarity:.2f}")
            plagiarized_pairs.append((names[i], names[j], similarity))

# Step 4: Visualize the similarity matrix
plt.figure(figsize=(10, 8))
df_sim = pd.DataFrame(similarity_matrix, index=names, columns=names)
sns.heatmap(df_sim, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Homework Similarity Heatmap")
plt.tight_layout()
plt.savefig("similarity_heatmap.png")
plt.show()

# Step 5: Export plagiarism results to a CSV file
results_df = pd.DataFrame(plagiarized_pairs, columns=["File 1", "File 2", "Similarity"])
results_df.to_csv("plagiarism_report.csv", index=False)
print("\n📄 Plagiarism report saved as 'plagiarism_report.csv'")
