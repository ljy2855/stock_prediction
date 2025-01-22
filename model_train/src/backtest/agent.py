import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import torch
import numpy as np


class BacktestAgent:
    def __init__(
        self, model, price_data, feature_data, date_data,
        n_steps=30, initial_balance=10000, signal_threshold=0.1,
        risk_tolerance=1.5, report_dir="reports", transaction_cost=0.0001
    ):
        """
        BacktestAgent 초기화.
        """
        self.model = model
        self.price_data = price_data
        self.feature_data = feature_data
        self.date_data = date_data
        self.n_steps = n_steps
        self.initial_balance = initial_balance
        self.signal_threshold = signal_threshold
        self.risk_tolerance = risk_tolerance
        self.report_dir = report_dir
        self.transaction_cost = transaction_cost
        self.results = None
        self.total_return = None
        self.final_portfolio_value = None
        self.trades = None

        os.makedirs(self.report_dir, exist_ok=True)

    def run(self):
        """
        백테스트 실행 메서드.
        """
        self.results, self.total_return, self.final_portfolio_value, self.trades = self._run_backtest()
        print("Backtest completed.")

    def _run_backtest(self):
        """
        내부 백테스트 로직.
        """
        balance = self.initial_balance
        position = 0
        daily_returns = []
        cumulative_balance = []
        signals = []
        predictions = []  # Transformer 모델의 30차원 벡터 저장
        trades = []

        for i in range(self.n_steps, len(self.price_data)):
            # 입력 데이터 생성
            input_sequence = (
                torch.tensor(self.feature_data[i - self.n_steps : i], dtype=torch.float32)
                .unsqueeze(0)  # Shape: (1, n_steps, num_features)
                .numpy()       # Convert to numpy array
            )
            
            # Transformer 모델 예측
            signal_value = self.model.predict(input_sequence)  # (1, 30)
            predictions.append(signal_value.flatten())  # Flatten to (30,)
            
            # 신호 생성
            signal = self._generate_signal(signal_value)
            signals.append(signal)
            
            price = self.price_data[i]

            # 매수/매도 로직
            if signal > 0 and position == 0:  # BUY
                position = balance / price
                balance = 0
                print(f"BUY: {self.date_data[i]} - Price: {price}")
                trades.append({"Date": self.date_data[i], "Price": price, "Type": "BUY"})

            elif signal < 0 and position > 0:  # SELL
                print(f"SELL: {self.date_data[i]} - Price: {price}")
                balance = position * price * (1 - self.transaction_cost)
                position = 0
                trades.append({"Date": self.date_data[i], "Price": price, "Type": "SELL"})

            portfolio_value = balance + position * price
            cumulative_balance.append(portfolio_value)

            daily_return = (portfolio_value - cumulative_balance[-2]) / cumulative_balance[-2] if len(cumulative_balance) > 1 else 0
            daily_returns.append(daily_return)

        # 결과 저장
        results = pd.DataFrame({
            "Date": self.date_data[self.n_steps:],
            "Price": self.price_data[self.n_steps:],
            "Signal": signals,
            "Prediction": predictions,  # Transformer 모델의 30차원 벡터 저장
            "Portfolio Value": cumulative_balance,
            "Daily Return": daily_returns,
        })

        trades_df = pd.DataFrame(trades)

        final_portfolio_value = cumulative_balance[-1] if cumulative_balance else balance
        total_return = (final_portfolio_value - self.initial_balance) / self.initial_balance

        return results, total_return, final_portfolio_value, trades_df


    def _generate_signal(self, signal_values):
        """
        신호 생성 메서드.
        """
        avg_signal = np.mean(signal_values)
        if avg_signal > self.signal_threshold:
            return 1  # 매수
        elif avg_signal < -self.signal_threshold:
            return -1  # 매도
        return 0  # 중립

    def save_report(self):
        """
        백테스트 리포트를 주피터 노트북(.ipynb) 및 Markdown(.md)으로 저장합니다.
        """
        report_path = os.path.join(self.report_dir, "backtest")
        os.makedirs(report_path, exist_ok=True)

        self._save_plot(report_path)
        self._save_parameters(report_path)
        self._save_trades(report_path)
        self._save_notebook(report_path)

        print(f"Report saved successfully at: {report_path}")

    def show_plot(self):
        """
        백테스트 결과 그래프를 시각화합니다.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        initial_price = self.results["Price"].iloc[0]
        scaled_portfolio = self.results["Portfolio Value"] / self.results["Portfolio Value"].iloc[0] * initial_price

        # 주가 그래프
        self.results["Date"] = pd.to_datetime(self.results["Date"])
        ax1.plot(self.results["Date"], self.results["Price"], label="Stock Price", color="orange", linewidth=2)
        ax1.plot(self.results["Date"], scaled_portfolio, label="Portfolio Value (Scaled)", color="blue", linewidth=2)

        # 거래 마커
        buy_label_shown = False
        sell_label_shown = False
        for _, trade in self.trades.iterrows():
            color = "green" if trade["Type"] == "BUY" else "red"
            marker = "^" if trade["Type"] == "BUY" else "v"
            if trade["Type"] == "BUY" and not buy_label_shown:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker, label="BUY")
                buy_label_shown = True
            elif trade["Type"] == "SELL" and not sell_label_shown:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker, label="SELL")
                sell_label_shown = True
            else:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker)


        ax1.set_title("Backtest Results")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Value")
        ax1.legend()
        ax1.grid()

        plt.tight_layout()
        plt.show()

    def _save_plot(self, report_path):
        """
        그래프 저장.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        initial_price = self.results["Price"].iloc[0]
        scaled_portfolio = self.results["Portfolio Value"] / self.results["Portfolio Value"].iloc[0] * initial_price

        # 주가 그래프
        self.results["Date"] = pd.to_datetime(self.results["Date"])
        ax1.plot(self.results["Date"], self.results["Price"], label="Stock Price", color="orange", linewidth=2)
        ax1.plot(self.results["Date"], scaled_portfolio, label="Portfolio Value (Scaled)", color="blue", linewidth=2)

        # 거래 마커
        buy_label_shown = False
        sell_label_shown = False
        for _, trade in self.trades.iterrows():
            color = "green" if trade["Type"] == "BUY" else "red"
            marker = "^" if trade["Type"] == "BUY" else "v"
            if trade["Type"] == "BUY" and not buy_label_shown:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker, label="BUY")
                buy_label_shown = True
            elif trade["Type"] == "SELL" and not sell_label_shown:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker, label="SELL")
                sell_label_shown = True
            else:
                ax1.scatter(trade["Date"], trade["Price"], color=color, s=50, marker=marker)


        ax1.set_title("Backtest Results")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Value")
        ax1.legend()
        ax1.grid()

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, "backtest_plot.png"))
        plt.close()

    def _save_notebook(self, report_path):
        """
        주피터 노트북(.ipynb)로 리포트 저장.
        """
        nb = new_notebook()

        # Markdown 설명
        nb.cells.append(new_markdown_cell("# Backtest Report"))
        nb.cells.append(new_markdown_cell("## 1. Parameters"))
        nb.cells.append(new_code_cell(json.dumps({
            "Initial Balance": self.initial_balance,
            "Signal Threshold": self.signal_threshold,
            "Risk Tolerance": self.risk_tolerance,
            "Final Portfolio Value": self.final_portfolio_value,
            "Total Return": self.total_return
        }, indent=4)))

        # 그래프
        nb.cells.append(new_markdown_cell("## 2. Performance Graph"))
        nb.cells.append(new_code_cell("from PIL import Image\nimg = Image.open('backtest_plot.png')\nimg.show()"))

        # 거래 정보
        nb.cells.append(new_markdown_cell("## 3. Trade Log"))
        nb.cells.append(new_code_cell("import pandas as pd\ntrades = pd.read_csv('trades.csv')\nprint(trades.head())"))

        with open(os.path.join(report_path, "backtest_report.ipynb"), "w") as f:
            nbformat.write(nb, f)

    def _save_parameters(self, report_path):
        """
        사용된 파라미터 및 최종 결과 저장.
        """
        report = {
            "Initial Balance": self.initial_balance,
            "Signal Threshold": self.signal_threshold,
            "Risk Tolerance": self.risk_tolerance,
            "Final Portfolio Value": self.final_portfolio_value,
            "Total Return": self.total_return,
        }
        with open(os.path.join(report_path, "parameters.json"), "w") as f:
            json.dump(report, f, indent=4)

    def _save_trades(self, report_path):
        """
        거래 정보 저장.
        """
        self.trades.to_csv(os.path.join(report_path, "trades.csv"), index=False)