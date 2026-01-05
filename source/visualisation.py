import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_xy(
    x,
    y,
    plot_type: str= "line",
    title: str  = "",
    xlabel: str = "",
    ylabel: str = "",
    regression_line: bool = False
):
    
    plt.figure(figsize=(10, 5))

    if plot_type == "line":
        plt.plot(x,y)

    elif plot_type == "scatter":
        plt.scatter(x,y)
        if regression_line:
            X = x.values.reshape(-1, 1)
            model = LinearRegression()
            y_regression = model.fit(X, y.values).predict(X)
            plt.plot(x, y_regression, color='red', linewidth=2)

    elif plot_type == "bar":
        plt.bar(x,y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_type != "bar":
        plt.grid(True)
    plt.show()
