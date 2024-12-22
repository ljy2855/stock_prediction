import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.dates as mdates

def backtest(
    model, price_data, feature_data, date_data, n_steps, initial_balance=10000, transaction_cost=0.001
):
    """
    백테스트 함수 (벡터 출력 기반).
    :param model: Transformer 모델 (inference 메서드 포함)
    :param price_data: 테스트 데이터의 종가 데이터 (numpy 배열 또는 pandas Series)
    :param feature_data: 입력 데이터 (numpy 배열 또는 pandas DataFrame)
    :param date_data: 날짜 데이터 (numpy 배열 또는 pandas Series)
    :param n_steps: 모델 입력 시퀀스 길이
    :param initial_balance: 초기 자본
    :param transaction_cost: 거래 비용 비율 (예: 0.001은 0.1%)
    :return: 백테스트 결과 DataFrame, 최종 수익률, 총 자산 가치, 거래 정보
    """
    # 입력 데이터 유효성 검사
    if len(price_data) != len(feature_data) or len(price_data) != len(date_data):
        raise ValueError("price_data, feature_data, date_data의 길이가 일치하지 않습니다.")
    if len(price_data) <= n_steps:
        raise ValueError("price_data의 길이는 n_steps보다 커야 합니다.")
    
    # 초기화
    balance = initial_balance
    position = 0  # 현재 포지션 (보유 주식 수)
    daily_returns = []  # 일별 수익률
    cumulative_balance = []  # 누적 잔고
    signals = []  # 매수/매도 신호 저장
    trades = []  # 거래 정보 저장

    # 백테스트 루프
    for i in range(n_steps, len(price_data)):
        try:
            # 입력 시퀀스 생성
            input_sequence = torch.tensor(feature_data[i-n_steps:i], dtype=torch.float32).unsqueeze(0)
            
            # 모델 예측 (inference)
            with torch.no_grad():
                signal_values = model(input_sequence).cpu().numpy().flatten()
                
            
            # 매수/매도 신호 생성
            normalized_values = (signal_values - np.mean(signal_values)) / (np.std(signal_values) + 1e-9)  # 정규화
            scaled_values = normalized_values * 1e6  # 스케일 확대
            avg_signal = np.mean(scaled_values)  # 평균값 계산

            # 신호 생성 기준 조정
            signal = 1 if avg_signal > 0.02 else -1 if avg_signal < -0.02 else 0  # 기준에 따라 신호 결정
        except Exception as e:
            print(f"Error during inference at index {i}: {e}")
            signals.append(0)  # 신호 생성 실패 시 중립 유지
            continue

        signals.append(signal)
        price = price_data[i]  # 현재 종가

        # 거래 수행
        if signal > 0 and position == 0:  # 매수 신호
            position = balance / price  # 현재 자본으로 최대 구매
            balance = 0  # 자본은 모두 주식 구매로 사용
            trades.append({
                "Date": date_data[i],
                "Price": price,
                "Position": position,
                "Type": "BUY",
            })
        elif signal < 0 and position > 0:  # 매도 신호
            balance = position * price * (1 - transaction_cost)  # 주식 매도 후 잔고
            trades.append({
                "Date": date_data[i],
                "Price": price,
                "Position": position,
                "Type": "SELL",
            })
            position = 0

        # 현재 포트폴리오 가치 계산
        portfolio_value = balance + position * price
        cumulative_balance.append(portfolio_value)

        # 일별 수익률 계산
        if len(cumulative_balance) > 1:
            daily_return = (
                (portfolio_value - cumulative_balance[-2]) / cumulative_balance[-2]
            )
        else:
            daily_return = 0
        daily_returns.append(daily_return)

    # 백테스트 결과 저장
    results = pd.DataFrame(
        {
            "Date": date_data[n_steps:],
            "Price": price_data[n_steps:],
            "Signal": signals,
            "Portfolio Value": cumulative_balance,
            "Daily Return": daily_returns,
        }
    )

    # 거래 정보 저장
    trades_df = pd.DataFrame(trades)

    # 최종 결과 계산
    final_portfolio_value = cumulative_balance[-1] if cumulative_balance else initial_balance
    total_return = (final_portfolio_value - initial_balance) / initial_balance

    return results, total_return, final_portfolio_value, trades_df


def plot_backtest(results):
    """
    백테스트 결과 시각화 (포트폴리오 가치와 주가 가격을 정규화하여 비교).
    :param results: 백테스트 결과 DataFrame
    """
    plt.figure(figsize=(12, 6))
    
    # 정규화: 처음 값을 기준으로 나누어 스케일을 1로 맞춤
    normalized_portfolio = results["Portfolio Value"] / results["Portfolio Value"].iloc[0]
    normalized_price = results["Price"] / results["Price"].iloc[0]
    
    # 정규화된 데이터 시각화
    plt.plot(results["Date"], normalized_portfolio, label="Normalized Portfolio Value", color="blue")
    plt.plot(results["Date"], normalized_price, label="Normalized Stock Price", color="orange")
    
    # x축 날짜 포맷 간략화 (연도별 표시)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # 연도별 표시
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # 연도 포맷
    
    # 시각화 설정
    plt.title("Backtest Result: Portfolio vs. Stock Price (Normalized)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()  # 레이아웃 정리
    plt.show()
