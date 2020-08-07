import matplotlib.pyplot as plt


def plot(*args):
    size = len(args)
    fig, axes = plt.subplots(nrows=1, ncols=size, figsize=(15, 15))

    for i, arg in enumerate(args):
        ax = axes[i] if size > 1 else axes
        im = ax.imshow(arg)
        fig.colorbar(im, ax=ax, shrink=0.8 / size)

    plt.show()


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get("padder", 0)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value


def summary(*args):
    print(
        "\n".join(
            f"{k+1}) {dt}, {er}, {hd} " for k, (dt, er, hd) in enumerate(zip(*args))
        )
    )
