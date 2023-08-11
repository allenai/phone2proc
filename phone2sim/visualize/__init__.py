import matplotlib.pyplot as plt
import numpy as np


def draw_scene(line_segments, objects, object_rects) -> plt.Figure:
    # DEBUG: draw the object rectangles
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, ((x1, z1), (x2, z2)) in enumerate(line_segments):
        ax.plot([x1, x2], [z1, z2], label=f"wall {i}")
    for rect in object_rects:
        ax.plot(
            [rect[0][0], rect[1][0], rect[2][0], rect[3][0], rect[0][0]],
            [rect[0][1], rect[1][1], rect[2][1], rect[3][1], rect[0][1]],
            label="object",
        )
    for obj in objects:
        rect_mean_x = sum(p[0] for p in obj["top_down_rect"]) / 4
        rect_mean_z = sum(p[1] for p in obj["top_down_rect"]) / 4

        # draw an arrow in the direction of obj["rotation"]
        ax.arrow(
            rect_mean_x,
            rect_mean_z,
            -0.5 * np.cos((obj["rotation"] + 90) * np.pi / 180),
            0.5 * np.sin((obj["rotation"] + 90) * np.pi / 180),
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
        )
        ax.text(rect_mean_x, rect_mean_z, obj["object_type"], ha="center", va="center")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.axis("equal")
    return fig


def visualize_line_segments(line_segments, house):
    fig, ax = plt.subplots(figsize=(15, 10))

    if line_segments is not None:
        for i, ((x1, z1), (x2, z2)) in enumerate(line_segments):
            # color each line segment differently
            colors = np.random.rand(3)
            ax.arrow(
                x1,
                z1,
                x2 - x1,
                z2 - z1,
                head_width=0.1,
                head_length=0.1,
                fc=colors,
                ec=colors,
            )

            # LARGE FONT
            # annotate each line segment with its index
            ax.annotate(
                str(i),
                xy=((x1 + x2) / 2, (z1 + z2) / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=20,
            )

    if house is not None:
        for wall in house["walls"]:
            p1 = wall["polygon"][0]
            p2 = wall["polygon"][2]
            ax.arrow(
                p1["x"],
                p1["z"],
                p2["x"] - p1["x"],
                p2["z"] - p1["z"],
                head_width=0.1,
                head_length=0.1,
                fc="k",
                ec="k",
            )

    ax.axis("equal")
    fig.savefig("temp5.png")

    return fig
