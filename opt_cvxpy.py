from datetime import datetime, timedelta

import cvxpy as cp
import numpy as np


def optimize_401k(
    base_salary: int | float,
    raise_rate: float,
    bonus_rate: float,
    max_contrib: int | int,
    match_rate: float,
    growth_rate: float,
):
    periods = 26
    salaries = np.full(periods, base_salary / periods)
    salaries[5:] *= (1 + raise_rate)
    salaries[15:] *= (1 + raise_rate)
    salaries[5] += base_salary * bonus_rate
    total_income = np.sum(salaries)

    contribution_rate = cp.Variable(shape=periods - 1, integer=True)
    final_contribution = cp.Variable(nonneg=True)
    match_contrib = cp.Variable(shape=periods, nonneg=True)
    balance = cp.Variable(nonneg=True)
    true_up = cp.Variable(nonneg=True)

    total_contributed = cp.sum(cp.multiply(contribution_rate / 100, salaries[:-1]))
    total_contributed += final_contribution
    max_match = match_rate * total_income
    total_match = cp.sum(match_contrib)

    constraints = [
        contribution_rate >= 0,
        contribution_rate <= 50,
        final_contribution <= 0.5 * salaries[-1],
        total_contributed <= max_contrib,
        match_contrib <= match_rate * salaries,
        match_contrib <= (contribution_rate / 100) @ salaries[:-1],
        match_contrib[-1] <= final_contribution,
        true_up == max_match - total_match,
        balance == (total_contributed + total_match) * (1 + growth_rate) + true_up,
    ]

    objective = cp.Maximize(balance)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP)

    print(f"Total Income: {total_income:,.2f}")
    print(f"Total Contributed: {total_contributed.value:,.2f}")
    print(f"Total Matched: {total_match.value:,.2f}")
    print(f"Final True-up: {true_up.value:,.2f}")
    print(f"Final Balance: {balance.value:,.2f}")
    print(f"Final Contribution (Last Period): ${final_contribution.value:,.2f}")

    if prob.status == cp.OPTIMAL:
        return (
            prob.value,
            np.append(contribution_rate.value, final_contribution.value / salaries[-1]),
        )
    else:
        em = f"Solver status: {prob.status}"
        raise Exception(em)


def print_results(obj, x):
    print(f"Ending balance: ${obj:,.2f}")
    start_date = datetime(2025, 1, 10)
    pay_dates = [
        (start_date + timedelta(weeks=2 * p)).strftime("%Y-%m-%d")
        for p in range(26)
    ]
    for date, rate in zip(pay_dates, x / 100, strict=True):
        print(f"{date}: {rate:.1%}")


if __name__ == "__main__":
    obj, x = optimize_401k(
        base_salary=100_000,
        raise_rate=0.01,
        bonus_rate=0.10,
        max_contrib=23_500,
        match_rate=0.04,
        growth_rate=0.0,
    )
    print_results(obj, x)
