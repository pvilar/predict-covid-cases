# Logistic Growth Model for estimating Corona Virus spread

:construction: Model continuously being improved :construction:

The purpose of this model is to estimate the duration and the total number of cases of the [COVID-19](https://www.who.int/emergencies/diseases/novel-coronavirus-2019/) spread by country.

## Data source
The model reads data from the [data repository](https://github.com/CSSEGISandData/COVID-19) for the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) 

## How to use
1. Clone the repo to your local machine and install the package:
```bash
# go to the repo directory and run from the terminal:
pip install -e .
```
2. Import the functions:
```python
# in Python
from data import corona.read_covid_cases
from models import corona.LogisticCurve
```
3. Download data for a specific country:
```python
df = read_covid_cases(countries=["ITALY"])
```
4. Fit the Logistic Growth Model
```python
# select "confirmed" to predict cases or "deaths" for fatalities
y = df["confirmed"].values
lc = LogisticCurve()
lc.fit(y)
```
5. Run predictions
```python
y_pred, (y_pred_upper, y_pred_lower) = lc.predict(y)
```
6. Plot results
```python
lc.plot(y)
```
## Output example
<p align="center">
    <img src="italy prediction.png" width=850>
</p>

## Contact us
For further information or questions reach out to:

* Sebastian Vermaas (sebastian.vermaas@nike.com)
* Rob Hovens (Rob.Hovens@nike.com)
* Pau Vilar (pau.vilar@nike.com)
