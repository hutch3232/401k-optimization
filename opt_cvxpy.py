from datetime import datetime, timedelta

import cvxpy as cp
import numpy as np


def optimize_401k(
        base_salary,
        raise_rate,
        bonus_rate,
        max_contrib,
        match_rate,
        growth_rate,
):
    periods = 26
    salaries = np.full(periods, base_salary / periods)
    salaries[5:] *= (1 + raise_rate)
    salaries[15:] *= (1 + raise_rate)
    salaries[5] += base_salary * bonus_rate

    contribution_rate = cp.Variable(shape=periods, integer=True)

    total_contributed = cp.sum(cp.multiply(contribution_rate / 100, salaries))

    balance = cp.Variable(shape=1)

    constraints = [balance == 0]
    for t in range(periods):
        balance = (
            balance
            * (1 + growth_rate / periods)
            + (contribution_rate[t] / 100 * salaries[t])
            + (match_rate * salaries[t])
        )

    objective = cp.Maximize(balance)

    constraints += [
        contribution_rate >= 6,
        contribution_rate <= 50,
        total_contributed <= max_contrib,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP)

    if prob.status == cp.OPTIMAL:
        return prob.value, contribution_rate.value
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
        growth_rate=0.05,
    )
    print_results(obj, x)
