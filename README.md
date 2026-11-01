# marketbot (Gold Lab)

Gold (XAU) ke liye ek **research aur backtesting tool**. Ye trading bot nahi hai, koi order nahi lagata.

## Ye kya karta hai
- Market snapshot: trend, RSI, volatility, support/resistance, dollar/yields correlation
- 2000+ strategy scenarios ka backtest (fees, slippage, leverage, liquidation ke saath)
- Train/test split taaki overfitting pakdi ja sake
- Monte Carlo aur fake price paths pe robustness test
- Leverage sweep (1x se 100x)

## Install
```
pip install numpy pandas matplotlib yfinance
```

## Chalane ka tareeka
```
python gold_lab.py                                  # real gold data (yfinance)
python gold_lab.py --source csv --csv mydata.csv    # TradingView CSV
python gold_lab.py --source synth                   # random data, sanity check
python gold_lab.py --lev 10 --capital 100           # custom settings
```
Output: `gold_lab_results.csv` aur `gold_lab_report.png`

## Disclaimer
Sirf educational use ke liye. Backtest ka matlab future me wahi result nahi hota.
Ye financial advice nahi hai. High leverage me poora paisa kam samay me khatam ho sakta hai.

Code Claude (Anthropic) ki madad se banaya gaya hai.
