#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "fire"]
# ///


ideal = {
    0: 1.61,
    1: 0.81,
    2: 0.54,
    3: 0.40,
    4: 0.32,
    5: 0.27,
}

post = {
    0: 1.61,
    1: 0.86,
    2: 0.57,
    3: 0.43,
    4: 0.34,
    5: 0.29,
}

pre = {}


def plot():
    import matplotlib.pyplot as plt
    plt.plot(list(map(lambda x: x +1, ideal.keys())), list(ideal.values()), label="Ideal")
    plt.plot(list(map(lambda x: x +1, pre.keys())), list(pre.values()), label="Pre-fix")
    plt.plot(list(map(lambda x: x +1, post.keys())), list(post.values()), label="Post-fix")
    plt.xlabel("Epoch")
    plt.ylabel("it/s")
    plt.yscale("log")
    plt.title("Training Speed (it/s) vs Rollout Length (Epoch - 1)")
    plt.legend()
    plt.grid()
    plt.savefig("rollout.png")

if __name__ == "__main__":
    import fire
    fire.Fire(plot)