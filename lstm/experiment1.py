import datetime as dt
import time as tm
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def pr(open, close, trend):
    if open < close:
        return trend == 'bull'
    elif open > close:
        return trend == 'bear'


def process(row):
    bull_break = row['Close'] > row['c1'] and row['Close'] > row['c1']
    bear_break = row['Close'] < row['c1'] and row['Close'] < row['c2']

    if pr(row['o1'], row['c1'], 'bull') and pr(row['o2'], row['c2'], 'bull') and pr(row['o3'], row['c3'], 'bull') and bull_break:
        # three white soldier
        return 1
    if pr(row['o1'], row['c1'], 'bear') and pr(row['o2'], row['c2'], 'bear') and pr(row['o3'], row['c3'], 'bear') and bear_break:
        # three dead crow
        return -1
    if pr(row['o3'], row['c3'], 'bear') and pr(row['o2'], row['c2'], 'bull') and pr(row['o1'], row['c1'], 'bull') and bull_break:
        # three ghg
        return 1
    else:
        return 0


if __name__ == '__main__':
    STOCK = 'GC=F'

    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = (dt.date.today() - dt.timedelta(days=50)).strftime('%Y-%m-%d')
    df = yf.download(tickers=STOCK, start=date_3_years_back, end=date_now, interval="15m", auto_adjust=False,
                     prepost=False)

    df = df.drop(['High', 'Low', 'Adj Close'], axis=1)

    df['o1'] = df['Open'].shift(1)
    df['c1'] = df['Close'].shift(1)
    df['o2'] = df['Open'].shift(2)
    df['c2'] = df['Close'].shift(2)
    df['o3'] = df['Open'].shift(3)
    df['c3'] = df['Close'].shift(3)

    raw_seq = df[['Close', 'Open', 'o1', 'c1', 'o2', 'c2', 'o3', 'c3']].dropna()
    _action = pd.DataFrame(index=raw_seq.index, columns=['action'])
    for index, row in raw_seq.iterrows():
        next = process(row)
        _action.at[index, 'action'] = next
    raw_seq['action'] = _action

    features = raw_seq[['o1', 'c1', 'o2', 'c2', 'o3', 'c3']]
    actions = raw_seq['action']

    X_train, X_test, y_train, y_test = train_test_split(features, actions, test_size=0.2, random_state=42)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    # Print the R-squared score
    print("R-squared score:", r2 * 100)
