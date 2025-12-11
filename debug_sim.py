import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Vector A
a = np.array([[1, 0, 0]])
# Vector B (identical)
b = np.array([[1, 0, 0]])
# Vector C (opposite)
c = np.array([[-1, 0, 0]])
# Vector D (orthogonal)
d = np.array([[0, 1, 0]])

search = np.vstack([b, c, d])

sims = cosine_similarity(a, search)[0]
print(f"Similarities: {sims}")
print(f"Argmax: {np.argmax(sims)}")
print(f"Expected Argmax: 0 (index of b)")

# Check if argmax picks the most similar
if np.argmax(sims) == 0:
    print("Argmax picks the most similar (closest to 1).")
else:
    print("Argmax does NOT pick the most similar.")
