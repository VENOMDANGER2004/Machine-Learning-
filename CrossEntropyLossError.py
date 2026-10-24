import math


def cross_entropy(input_data):
    loss = 0
    n = len(input_data)

    for w_sum, y in input_data:
        # Clip w_sum to avoid log(0)
        w_sum = min(max(w_sum, 1e-15), 1 - 1e-15)

        # Use natural log consistently
        term = -(y * math.log(w_sum) + (1 - y) * math.log(1 - w_sum))
        print(f"Loss for entry (w_sum={w_sum}, y={y}): {term}")
        loss += term

    return loss / n


# Example input: list of tuples (predicted probability, true label)
input_data = [(0.9, 1), (0.2, 0), (0.8, 1), (0.1, 0)]

error_term = cross_entropy(input_data)
print(f"\nAverage cross-entropy loss: {error_term}")