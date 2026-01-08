import json
import matplotlib.pyplot as plt

filename = "5_anchors_C.json"
with open(filename, "r") as f:
    data = json.load(f)

anchors = []
anchor_labels = []
for anchor_id, pos in data["anchor_positions"].items():
    anchors.append(tuple(pos))
    anchor_labels.append(anchor_id)

x_vals = [point["x_cm"] for point in data["position_data"]]
y_vals = [point["y_cm"] for point in data["position_data"]]

avg_x = sum(x_vals) / len(x_vals)
avg_y = sum(y_vals) / len(y_vals)

plt.figure(figsize=(7, 7))

plt.scatter(
    x_vals, y_vals,
    s=40, alpha=0.8,
    label="Estimated positions"
)

plt.scatter(
    avg_x, avg_y,
    c="red", marker="x", s=120,
    label=f"Average position ({avg_x:.1f}, {avg_y:.1f})"
)

ax, ay = zip(*anchors)
plt.scatter(
    ax, ay,
    c="blue", marker="^", s=100,
    label="Anchors"
)

for anchor_id, (x, y) in zip(anchor_labels, anchors):
    plt.text(x + 5, y + 5, anchor_id, fontsize=9, color="blue")

plt.xlim(avg_x - 50, avg_x + 200)
plt.ylim(avg_y - 50, avg_y + 200)

plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.title("Localization Estimates with Anchor Positions")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

