import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(
    rc={
        "grid.alpha": 1.0,
        "grid.color": "black",
        "grid.linestyle": "-",
        "grid.linewidth": 0.1,
        "axes.facecolor": "white",
        "axes.titlesize": 20.0,
        "axes.labelsize": 14.0,
        "font.sans-serif": ["Futura", "sans-serif"],
        "figure.figsize": [14.0, 6.0],
        "legend.shadow": False,
    }
)
sns.set_palette(["#16315a", "#ff4800", "#c4d82e", "#000000", "#8C8C8C", "#27AAE1"])


class LogisticCurve:
    def __init__(self, parameters: tuple = (1, 1, 100)):

        self.fit_errors = None
        self.duration = None
        self.total_cases = None

        self.optimized_parameters = None
        self.duration_errors = None
        self.total_cases_errors = None
        self.optimized_parameters = None
        self.parameters = parameters

        self.exponential_decay_model = lambda x, C, k: C * np.exp((-k * x))
        self.logistic_model = lambda x, a, b, c: c / (1 + np.exp(-(x - b) / a))

    def fit(self, y: np.array):

        X = np.arange(0, len(y))

        optimized_fit = scipy.optimize.curve_fit(
            self.logistic_model,
            X,
            y,
            self.parameters,
            maxfev=100000,
            bounds=(0, [50, 50, 300000]),
        )

        self.optimized_parameters = optimized_fit[0]

        (
            duration_predictions,
            total_cases_predictions,
        ) = self._get_predictions_recursively(y)

        self.duration_errors = self._get_error_curve(duration_predictions)
        self.total_cases_errors = self._get_error_curve(total_cases_predictions)
        self.fit_errors = [np.sqrt(optimized_fit[1][i][i]) for i in [0, 1, 2]]

    def predict(self, y: np.array):
        a, b, c = self.optimized_parameters

        if None in (a, b, c):
            raise ValueError("You must fit this model first! Call the .fit() method")

        X = np.arange(0, len(y)).tolist()

        self.total_cases = np.round(c, 0)
        self.duration = int(
            scipy.optimize.fsolve(lambda x: self.logistic_model(x, a, b, c) - int(c), b)
        )

        X_pred = np.concatenate([X, np.arange(max(X) + 1, self.duration)])
        y_pred = [self.logistic_model(i, a, b, c) for i in X_pred]
        y_pred = np.array(y_pred)

        duration_upper = self.duration + (
            self.duration * self.duration_errors[-1]
        )  # self.fit_errors[0])
        total_cases_upper = c + (c * self.total_cases_errors[-1])
        X_pred_upper = np.concatenate([X, np.arange(max(X) + 1, duration_upper)])
        y_pred_upper = [
            self.logistic_model(i, a, b, total_cases_upper) for i in X_pred_upper
        ]

        y_pred_upper = np.concatenate([y_pred[: len(y)], y_pred_upper[len(y) + 1 :]])

        duration_lower = self.duration - (self.duration * self.fit_errors[0])

        total_cases_lower = c - (c * (self.fit_errors[2] / c))
        X_pred_lower = np.concatenate([X, np.arange(max(X) + 1, duration_lower)])
        y_pred_lower = [
            self.logistic_model(i, a, b, total_cases_lower) for i in X_pred_lower
        ]
        y_pred_lower = np.concatenate([y_pred[: len(y)], y_pred_lower[len(y) + 1 :]])

        return y_pred, (y_pred_upper, y_pred_lower)

    def _get_predictions_recursively(self, y):

        duration_predictions = list()
        total_cases_predictions = list()

        for i in range(1, len(y) + 1):

            y_ = y[:i]
            X_ = np.arange(0, len(y_))

            optimized_fit = scipy.optimize.curve_fit(
                self.logistic_model,
                X_,
                y_,
                (1, 1, 100),
                maxfev=100000,
                bounds=(0, [50, 50, 300000]),
            )

            a, b, c = optimized_fit[0]

            duration = int(
                scipy.optimize.fsolve(
                    lambda x: self.logistic_model(x, a, b, c) - int(c), b
                )
            )

            total_cases = np.round(c, 0)

            duration_predictions.append(duration)
            total_cases_predictions.append(total_cases)

        return duration_predictions, total_cases_predictions

    def _get_error_curve(self, predictions):
        y_mean = [predictions[0]]
        for i in range(2, len(predictions) + 1):
            w = list([x ** 3 for x in range(1, len(predictions[:i]) + 1)])
            m = np.average(list(predictions[:i]), weights=w)
            y_mean.append(m)

        # calculate error at each time step
        real_value = np.mean(y_mean[-1])
        error = (predictions - real_value) / real_value

        # fit an exponential decay curve
        error_fit, error_cov = scipy.optimize.curve_fit(
            self.exponential_decay_model,
            range(len(error)),
            abs(error),
            [1, 0.1],
            maxfev=10000,
        )
        tdata = np.arange(0, len(error))
        error_curve = [
            self.exponential_decay_model(i, error_fit[0], error_fit[1]) for i in tdata
        ]
        return error_curve

    def plot(self, y, title: str = "", ylab: str = ""):
        self.fit(y)

        y_pred, (y_pred_upper, y_pred_lower) = self.predict(y)

        lower_duration = len(y_pred_lower)
        upper_duration = len(y_pred_upper)

        y_pred = np.concatenate(
            [y_pred, [max(y_pred)] * (upper_duration - len(y_pred))]
        )
        y_pred_lower = np.concatenate(
            [y_pred_lower, [max(y_pred_lower)] * (upper_duration - len(y_pred_lower))]
        )

        title += (
            "\nEstimated duration of virus spread from today: "
            + str(lower_duration - len(y))
            + " to "
            + str(upper_duration - len(y))
            + " days"
        )
        ylab += ("Number of total cases")
        plt.title(title)
        plt.plot(range(len(y_pred)), y_pred, label="Predicted")
        plt.scatter(range(len(y)), y, label="Actuals")
        plt.fill_between(
            range(len(y_pred_upper)),
            y_pred_lower,
            y_pred_upper,
            color=sns.color_palette()[4],
            alpha=0.15,
            label="Confidence Interval",
        )
        plt.ylabel(ylab)
        plt.xlabel('Days from first infection case')

        plt.legend()
        plt.show()
